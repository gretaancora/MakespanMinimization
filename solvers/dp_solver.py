"""
Programmazione dinamica pseudo-polinomiale per Pm||Cmax (dp_p_m_cmax.md).

Stato: R_i = insieme dei vettori di carico (x_1,...,x_{m-1}) raggiungibili
sulle prime m-1 macchine usando i primi i job. Il carico della macchina m e'
implicito (= somma totale dei primi i job - somma delle altre m-1).

--------------------------------------------------------------------------
DUE RAPPRESENTAZIONI, SCELTE AUTOMATICAMENTE
--------------------------------------------------------------------------
"sparsa" (dims = m-1 qualunque): R_i tenuto come Python `set` di tuple,
fedele alla definizione del file di riferimento. Lo spazio degli stati
effettivamente raggiungibili e' spesso molto piu' piccolo del bound teorico
(P+1)^dims, quindi questa rappresentazione resta la scelta giusta quando
dims e' grande (m>=4): un array denso di dimensione (P+1)^dims sarebbe li'
quasi sempre troppo grande per stare in memoria. Il costo e' che l'espansione
degli stati e' un loop Python job per job, stato per stato -- lento su
istanze dove il numero di stati raggiunge decine/centinaia di milioni.

"densa" (dims in {1,2}, cioe' m in {2,3}): R_i tenuto come array booleano
numpy di forma (P+1,)*dims, dove reachable[x] = True se lo stato x e'
raggiungibile. La transizione (assegna il job a una macchina esplicita h,
oppure lascialo alla macchina implicita) diventa uno shift + OR vettoriale
per asse invece di un loop Python su ogni stato -- le stesse operazioni,
eseguite pero' in C da numpy invece che dall'interprete Python. Per dims<=2
questo e' quasi sempre molto piu' veloce della versione sparsa (anche 10-50x
su istanze dove prima si arrivava a decine di minuti), a patto che
(P+1)^dims stia comodamente in memoria -- motivo per cui la si usa solo per
dims piccoli: gia' a dims=4 (m=5), (P+1)^4 esplode per qualunque P non
banale, quindi la` la scelta densa non verrebbe comunque mai selezionata.
La selezione tra le due e' automatica (vedi `_use_dense`), in base a una
stima di quanto spazio occuperebbe la versione densa rispetto al budget di
memoria dato. Per calibrazione/benchmark, `solve_dp(..., force_representation=
"dense"|"sparse")` permette di bypassare questa stima e imporre l'una o
l'altra esplicitamente sulla stessa istanza (vedi calibrate_dp_lpt.py); il
limite di memoria reale (RLIMIT_AS) resta comunque attivo in entrambi i casi.

Limite della rappresentazione densa: non supporta la ricostruzione esplicita
dell'assegnamento (richiederebbe uno storico di array (P+1)^dims, uno per
job, insostenibile). Se viene richiesta (`reconstruct_assignation=True`) si
usa sempre la rappresentazione sparsa, indipendentemente da quanto
converrebbe quella densa in termini di velocita'.

--------------------------------------------------------------------------
GESTIONE DELLA MEMORIA E INCUMBENT
--------------------------------------------------------------------------
Il calcolo gira in un sottoprocesso dedicato con un limite reale di memoria
virtuale a livello di OS (RLIMIT_AS): se lo si supera (rappresentazione
sparsa O densa, numpy solleva MemoryError esattamente come una allocazione
Python quando l'allocatore sottostante fallisce), lo si intercetta invece di
un crash non gestito. In entrambe le rappresentazioni si tiene traccia
dell'ultimo livello di stati completato con successo, e da li' si estrae
l'incumbent: il cmax del miglior stato raggiungibile usando SOLO i primi
`n_jobs_done` job (quelli su cui la DP ha effettivamente completato
l'espansione prima di esaurire la memoria). Non si tenta alcun
completamento dei job restanti (ne' greedy LPT ne' altro): la DP si ferma
li', esplicitamente. Di conseguenza questo cmax NON e' un upper bound
ammissibile per l'istanza intera (i job da n_jobs_done a n restano non
assegnati) -- e' solo il punto in cui il calcolo esatto si e' interrotto.
Se anche costruire questo incumbent parziale esaurisse la memoria residua,
si rinuncia (cmax=None) invece di rischiare un secondo crash.
"""

from __future__ import annotations
import gc
import mmap
import multiprocessing as mp
import resource
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass
class DPResult:
    status: str                    # "OPTIMAL" o "MEMORY_LIMIT_EXCEEDED"
    cmax: Optional[int]            # OPTIMAL: valore provato ottimo.
                                    # MEMORY_LIMIT_EXCEEDED: cmax del miglior stato DP
                                    # raggiunto sui primi n_jobs_dp_exact job (nessun
                                    # completamento dei job restanti), se disponibile,
                                    # altrimenti None. NON e' un upper bound ammissibile
                                    # per l'istanza intera.
    runtime: float
    n_states_final: int            # numero di stati raggiungibili a fine calcolo
                                    # (-1 se non noto/non calcolato)
    assignment: Optional[List[int]] = None
    assignment_note: Optional[str] = None
    n_jobs_dp_exact: Optional[int] = None  # quanti job sono stati trattati dalla DP prima
                                            # di esaurire la memoria (None se OPTIMAL = tutti gli n)
    representation: Optional[str] = None   # "dense" o "sparse", per trasparenza/debug
    solver_runtime: Optional[float] = None  # tempo speso DENTRO il sottoprocesso worker
                                             # (dal suo primo istante di esecuzione all'invio
                                             # sulla pipe): esclude fork(), join() e l'attesa
                                             # sulla pipe lato padre. None se non e' mai arrivata
                                             # risposta dal worker (vedi `runtime` per il totale).
    peak_memory_mb: Optional[float] = None  # picco di memoria RESIDENTE (RSS) del sottoprocesso
                                             # worker per QUESTA istanza (resource.getrusage(
                                             # RUSAGE_SELF).ru_maxrss letto poco prima dell'invio
                                             # sulla pipe). Dato che ogni chiamata a solve_dp()
                                             # lancia un sottoprocesso fork() NUOVO e dedicato,
                                             # questo valore e' specifico dell'istanza corrente e
                                             # non e' inquinato da chiamate precedenti (a
                                             # differenza di resource.getrusage(RUSAGE_CHILDREN)
                                             # nel padre, che accumula un massimo storico su tutti
                                             # i figli terminati). None se non e' mai arrivata
                                             # risposta dal worker.
    peak_vsz_mb: Optional[float] = None     # picco di memoria VIRTUALE (VmPeak da /proc/self/
                                             # status) dello stesso sottoprocesso worker, letto
                                             # nello stesso istante di peak_memory_mb. E' la
                                             # metrica su cui agisce RLIMIT_AS (vedi
                                             # _peak_vsz_mb): nelle righe MEMORY_LIMIT_EXCEEDED e'
                                             # quindi il valore vicino al budget --memory-limit-gb,
                                             # a differenza di peak_memory_mb (RSS) che e' tipica-
                                             # mente piu' basso. None se non disponibile (fuori da
                                             # Linux) o se non e' mai arrivata risposta dal worker.

    @property
    def is_optimal(self) -> bool:
        return self.status == "OPTIMAL"


# ---------------------------------------------------------------------------
# utilita' condivise
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    """Picco di memoria residente (RSS) del processo CHIAMANTE finora, in MB.
    Va richiamata DENTRO il sottoprocesso worker (RUSAGE_SELF e' per-processo):
    dato che _dp_worker gira in un fork() nuovo per ogni chiamata a solve_dp(),
    il valore restituito riflette solo il lavoro di QUESTA istanza. ru_maxrss e'
    riportato in kB su Linux (a differenza di macOS, dove sono byte: qui non
    serve gestirlo, il progetto gira in container Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _peak_vsz_mb() -> Optional[float]:
    """Picco di memoria VIRTUALE (VmPeak) del processo CHIAMANTE finora, in MB.
    E' la metrica su cui agisce davvero RLIMIT_AS (vedi _set_memory_limit): a
    differenza di _peak_rss_mb() (memoria residente/fisica), VmPeak include
    anche spazio di indirizzi riservato ma non necessariamente scritto --
    arene dell'allocatore non completamente popolate, librerie condivise
    (numpy/BLAS/LAPACK), frammentazione dovuta al rehashing dei `set` Python
    durante l'espansione della rappresentazione sparsa. Per questo tende a
    essere piu' alta della RSS a parita' di istante, ed e' la metrica giusta
    da confrontare col budget --memory-limit-gb quando si vuole vedere QUANTO
    VICINO al limite si e' arrivati (la RSS, essendo tipicamente piu' bassa,
    non lo mostra -- vedi discussione nel modulo di plotting). Va richiamata
    DENTRO il sottoprocesso worker, per lo stesso motivo di _peak_rss_mb().
    Letta da /proc/self/status (VmPeak e' in kB su Linux); se il file o il
    campo non sono disponibili (es. fuori da Linux, o kernel senza questo
    campo) restituisce None invece di sollevare un'eccezione, coerentemente
    con la tolleranza di _set_memory_limit."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    # formato tipico: "VmPeak:\t 1234567 kB\n"
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _set_memory_limit(memory_limit_bytes: int) -> None:
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            memory_limit_bytes = min(memory_limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))
    except (ValueError, OSError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# rappresentazione sparsa (set di tuple) -- usata per dims>=3, oppure quando
# serve ricostruire l'assegnamento esplicito, oppure quando la densa non
# ci starebbe in memoria
# ---------------------------------------------------------------------------

def _expand_sparse(R_prev: Set[Tuple[int, ...]], pi: int, dims: int) -> Set[Tuple[int, ...]]:
    R_cur = set(R_prev)  # opzione: job -> macchina m (stato invariato)
    for s in R_prev:
        for h in range(dims):
            new_s = list(s)
            new_s[h] += pi
            R_cur.add(tuple(new_s))
    return R_cur


def _reconstruct_sparse(p: List[int], dims: int, history: List[Set[Tuple[int, ...]]],
                         target_state: Tuple[int, ...]) -> List[int]:
    n = len(p)
    assignment = [-1] * n
    cur_state = list(target_state)
    for i in range(n - 1, -1, -1):
        pi = p[i]
        R_before = history[i]
        placed = False
        for h in range(dims):
            if cur_state[h] - pi >= 0:
                prev_state = cur_state.copy()
                prev_state[h] -= pi
                if tuple(prev_state) in R_before:
                    assignment[i] = h
                    cur_state = prev_state
                    placed = True
                    break
        if not placed:
            assignment[i] = dims  # macchina m (implicita)
    return assignment


def _best_state_sparse(R: Set[Tuple[int, ...]], P: int) -> Tuple[int, Tuple[int, ...]]:
    best_cmax, best_state = None, None
    for s in R:
        last_load = P - sum(s)
        c = max(max(s), last_load)
        if best_cmax is None or c < best_cmax:
            best_cmax, best_state = c, s
    return best_cmax, best_state


def _fallback_incumbent_sparse(p: List[int], n_jobs_done: int,
                                R_partial: Set[Tuple[int, ...]]) -> int:
    """Incumbent = cmax del miglior stato raggiungibile usando SOLO i primi
    n_jobs_done job. Nessun completamento dei job restanti: la DP si ferma
    qui (vedi nota nel docstring del modulo)."""
    P_partial = sum(p[:n_jobs_done])
    best_cmax, _ = _best_state_sparse(R_partial, P_partial)
    return best_cmax


# ---------------------------------------------------------------------------
# rappresentazione densa (array booleano numpy) -- usata per dims in {1,2}
# quando (P+1)^dims sta comodamente nel budget di memoria dato
# ---------------------------------------------------------------------------

def _use_dense(P: int, dims: int, memory_limit_bytes: int) -> bool:
    if dims == 0 or dims > 5:
        return False
    dense_bytes = (P + 1) ** dims  # 1 byte/elemento con dtype=bool di numpy
    # margine generoso: array principale + una copia temporanea per asse
    # durante ogni espansione + spazio per l'estrazione finale (submatrici
    # temporanee nella ricerca binaria per dims=2)
    return dense_bytes * 6 <= memory_limit_bytes


def choose_representation(P: int, dims: int, memory_limit_bytes: int) -> str:
    """Rappresentazione ("dense" o "sparse") che la modalita' automatica di
    `solve_dp` sceglierebbe per questi parametri, assumendo
    reconstruct_assignment=False (unico caso in cui la densa viene mai
    considerata -- vedi `solve_dp`). Esposta pubblicamente (a differenza di
    `_use_dense`) cosi' da poter essere richiamata da script di calibrazione
    esterni per confrontare la scelta automatica con quanto si osserva
    empiricamente forzando l'una o l'altra rappresentazione."""
    return "dense" if _use_dense(P, dims, memory_limit_bytes) else "sparse"


def _expand_dense(reachable: np.ndarray, pi: int, dims: int) -> np.ndarray:
    new_reachable = reachable.copy()  # opzione: job -> macchina implicita (stato invariato)
    for h in range(dims):
        size_h = reachable.shape[h]
        if pi >= size_h:
            continue  # spostarsi di pi lungo questo asse uscirebbe dal range [0,P]
        src = [slice(None)] * dims
        dst = [slice(None)] * dims
        src[h] = slice(0, size_h - pi)
        dst[h] = slice(pi, size_h)
        new_reachable[tuple(dst)] |= reachable[tuple(src)]
    return new_reachable


def _best_state_dense(reachable: np.ndarray, dims: int, P: int) -> Tuple[int, Tuple[int, ...]]:
    """Trova lo stato raggiungibile che minimizza il cmax, senza mai
    materializzare un array di costo grande quanto reachable (che con dtype
    numerico peserebbe piu' del booleano): per dims=1 basta un array 1D di
    al piu' P+1 indici validi; per dims=2 si usa una ricerca binaria sul
    valore obiettivo M, controllando la fattibilita' con maschere booleane
    al piu' grandi quanto reachable stesso.

    Nota: dims>=3 (raggiungibile solo forzando la densa con
    force_representation="dense" su m>=4 -- l'euristica automatica _use_dense
    non la sceglie mai li' in pratica) usa un fallback generico che
    materializza le coordinate degli stati True con np.argwhere: meno
    ottimizzato in memoria della ricerca binaria di dims=2, ma corretto per
    qualunque dims, a differenza del vecchio codice che applicava la
    logica di dims=2 indiscriminatamente a ogni dims!=1 (bug: per dims>=3
    il broadcasting tra la sotto-maschera (dims-dimensionale) e la maschera
    di costo r+c (2-dimensionale) falliva con un ValueError invece di dare
    un risultato scorretto silenzioso -- quindi era un fallimento rumoroso,
    non un'incongruenza silente, ma comunque un ValueError non gestito che
    la calibrazione con dense forzata su m=5 ha fatto emergere)."""
    if dims == 1:
        valid = np.flatnonzero(reachable)
        costs = np.maximum(valid, P - valid)
        best_idx = int(np.argmin(costs))
        return int(costs[best_idx]), (int(valid[best_idx]),)

    if dims == 2:
        def feasible(M: int) -> bool:
            if M >= P:
                return True
            sub = reachable[:M + 1, :M + 1]
            r = np.arange(M + 1)[:, None]
            c = np.arange(M + 1)[None, :]
            return bool(np.any(sub & ((r + c) >= (P - M))))

        lo, hi = 0, P
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid + 1
        M = lo
        sub = reachable[:M + 1, :M + 1]
        r = np.arange(M + 1)[:, None]
        c = np.arange(M + 1)[None, :]
        x1, x2 = np.argwhere(sub & ((r + c) >= (P - M)))[0]
        return M, (int(x1), int(x2))

    # dims >= 3: fallback generico (dims-agnostico), usato solo se qualcuno
    # forza la densa oltre dims=2 -- vedi nota sopra
    idx = np.argwhere(reachable)  # shape (K, dims), K = stati raggiungibili
    last_load = P - idx.sum(axis=1)
    costs = np.maximum(idx.max(axis=1), last_load)
    best_i = int(np.argmin(costs))
    return int(costs[best_i]), tuple(int(x) for x in idx[best_i])


def _fallback_incumbent_dense(p: List[int], n_jobs_done: int, reachable_partial: np.ndarray,
                               dims: int) -> int:
    """Incumbent = cmax del miglior stato raggiungibile usando SOLO i primi
    n_jobs_done job. Nessun completamento dei job restanti: la DP si ferma
    qui (vedi nota nel docstring del modulo)."""
    P_partial = sum(p[:n_jobs_done])
    best_cmax, _ = _best_state_dense(reachable_partial, dims, P_partial)
    return best_cmax


# ---------------------------------------------------------------------------
# worker (sottoprocesso)
# ---------------------------------------------------------------------------

def _dp_worker(p: List[int], dims: int, memory_limit_bytes: int,
                reconstruct_assignment: bool, force_representation: Optional[str], conn) -> None:
    t_worker_start = time.perf_counter()  # primo istante misurabile DENTRO il figlio:
                                           # esclude il costo di fork() (avvenuto prima che
                                           # questa riga venga eseguita) ma cattura tutto il
                                           # resto del lavoro del worker.
    n = len(p)
    P = sum(p)
    if force_representation == "dense":
        use_dense = True
    elif force_representation == "sparse":
        use_dense = False
    else:
        use_dense = (not reconstruct_assignment) and _use_dense(P, dims, memory_limit_bytes)

    n_jobs_done = 0
    reachable = None
    R_prev = None
    history = None
    memory_reserve = None

    try:
        _set_memory_limit(memory_limit_bytes)

        # Riserva di emergenza: allocata subito, DOPO aver fissato
        # RLIMIT_AS, cosi' "si ritaglia" una fetta del budget totale.
        # Se il calcolo principale va in MemoryError, la liberiamo per
        # prima cosa (vedi except sotto): senza questo margine, il
        # recupero (costruzione incumbent + conn.send) gira esattamente
        # al limite RLIMIT_AS e puo' a sua volta fallire con MemoryError,
        # perdendo silenziosamente l'incumbent (bug osservato: pipe
        # chiusa senza risposta, exitcode=0, dp_cmax vuoto).
        #
        # Usiamo mmap anonimo invece di bytearray: RLIMIT_AS conta lo
        # spazio di indirizzi VIRTUALE riservato, non le pagine
        # effettivamente scritte, quindi mmap.mmap(-1, N) "prenota" gli
        # stessi N byte di budget con una singola mmap() (microsecondi),
        # senza azzerare subito ~300+ MiB come faceva bytearray(N) --
        # che era un costo fisso di ~0.1s ad OGNI chiamata al worker,
        # confondibile con "tempo di calcolo" nelle misure di dp_solver_runtime
        # (individuato confrontando dp_solver_runtime su istanze n10/n50/n200
        # con m=2: era costante intorno a ~0.095-0.10s indipendentemente da n,
        # il che tradiva un costo fisso per chiamata, non rumore di sistema).
        reserve_bytes = min(max(int(memory_limit_bytes * 0.02), 32 * 1024 * 1024),
                             512 * 1024 * 1024)
        memory_reserve = mmap.mmap(-1, reserve_bytes)

        if use_dense:
            shape = tuple([P + 1] * dims)
            reachable = np.zeros(shape, dtype=bool)
            reachable[tuple([0] * dims)] = True
            for pi in p:
                reachable = _expand_dense(reachable, pi, dims)
                n_jobs_done += 1

            best_cmax, _ = _best_state_dense(reachable, dims, P)
            n_states = int(reachable.sum())
            conn.send(("OPTIMAL", best_cmax, n_states, None, "rappresentazione densa (numpy)",
                       n_jobs_done, "dense", time.perf_counter() - t_worker_start, _peak_rss_mb(),
                       _peak_vsz_mb()))
            return

        # -- rappresentazione sparsa --
        zero = tuple([0] * dims)
        R_prev = {zero}
        history = [R_prev] if reconstruct_assignment else None

        for pi in p:
            R_new = _expand_sparse(R_prev, pi, dims)
            if reconstruct_assignment:
                history.append(R_new)
            R_prev = R_new
            n_jobs_done += 1

        best_cmax, best_state = _best_state_sparse(R_prev, P)
        assignment = None
        if reconstruct_assignment:
            assignment = _reconstruct_sparse(p, dims, history, best_state)

        conn.send(("OPTIMAL", best_cmax, len(R_prev), assignment, None, n_jobs_done, "sparse",
                   time.perf_counter() - t_worker_start, _peak_rss_mb(), _peak_vsz_mb()))

    except MemoryError:
        # Prima cosa: liberare la riserva. Da qui in poi c'e' margine
        # reale di memoria per costruire l'incumbent e per inviarlo sulla
        # pipe, anche se il resto del processo era esattamente al limite
        # RLIMIT_AS un istante fa. Chiudiamo esplicitamente la mmap:
        # rilascia subito lo spazio di indirizzi virtuale, non aspettiamo
        # il garbage collector.
        if memory_reserve is not None:
            memory_reserve.close()
        del memory_reserve
        gc.collect()

        incumbent_cmax = None
        try:
            if use_dense and reachable is not None:
                incumbent_cmax = _fallback_incumbent_dense(p, n_jobs_done, reachable, dims)
            elif (not use_dense) and R_prev is not None:
                incumbent_cmax = _fallback_incumbent_sparse(p, n_jobs_done, R_prev)
        except MemoryError:
            incumbent_cmax = None

        del reachable, R_prev, history
        gc.collect()

        repr_label = "densa (numpy)" if use_dense else "sparsa"
        if incumbent_cmax is not None:
            note = (f"limite di memoria superato dopo {n_jobs_done}/{n} job "
                     f"(rappresentazione {repr_label}); cmax e' il miglior stato DP "
                     f"raggiunto sui primi {n_jobs_done} job, la DP si e' fermata li' "
                     f"(nessun completamento dei job restanti, NON e' un cmax "
                     f"ammissibile per l'istanza intera)")
        else:
            note = f"limite di memoria superato (rappresentazione {repr_label}), nessun incumbent disponibile"
        try:
            conn.send(("MEMORY_LIMIT_EXCEEDED", incumbent_cmax, -1, None, note, n_jobs_done,
                       "dense" if use_dense else "sparse", time.perf_counter() - t_worker_start,
                       _peak_rss_mb(), _peak_vsz_mb()))
        except MemoryError:
            pass  # il padre interpreta la pipe chiusa senza risposta come MEMORY_LIMIT_EXCEEDED
    except Exception as e:
        try:
            conn.send(("ERROR", None, -1, None, str(e), n_jobs_done, None,
                       time.perf_counter() - t_worker_start, _peak_rss_mb(), _peak_vsz_mb()))
        except Exception:
            pass
    finally:
        conn.close()


def solve_dp(processing_times: List[int], m: int,
             memory_limit_bytes: int = 16 * 1024**3,  # default: 16 GB
             reconstruct_assignment: bool = True,
             force_representation: Optional[str] = None) -> DPResult:
    """
    force_representation: None (default) -> scelta automatica (vedi
    `_use_dense` / `choose_representation`), come sempre. "dense" o "sparse"
    forzano esplicitamente l'una o l'altra, bypassando la stima automatica
    -- pensato per la calibrazione (vedi calibrate_dp_lpt.py), per misurare
    le due rappresentazioni sulla stessa istanza invece di lasciar decidere
    l'euristica. Il limite di memoria (memory_limit_bytes) resta comunque
    attivo e reale (RLIMIT_AS): forzare "dense" oltre cio' che l'euristica
    riterrebbe sicuro puo' quindi ancora finire in MEMORY_LIMIT_EXCEEDED,
    e questo e' voluto (e' proprio il modo in cui la calibrazione scopre
    dove sta il limite pratico della densa, non solo quello stimato).
    """
    if force_representation not in (None, "dense", "sparse"):
        raise ValueError("force_representation deve essere None, 'dense' o 'sparse' "
                          f"(ricevuto: {force_representation!r})")
    if force_representation == "dense" and reconstruct_assignment:
        raise ValueError("la rappresentazione densa non supporta reconstruct_assignment=True "
                          "(vedi docstring del modulo): usa force_representation='sparse', "
                          "force_representation=None, oppure reconstruct_assignment=False")

    start = time.perf_counter()
    p = processing_times
    n = len(p)
    dims = m - 1

    if dims == 0:
        runtime = time.perf_counter() - start
        return DPResult(status="OPTIMAL", cmax=sum(p), runtime=runtime,
                         n_states_final=1, assignment=[0] * n, n_jobs_dp_exact=n,
                         representation="trivial", solver_runtime=runtime,
                         peak_memory_mb=None, peak_vsz_mb=None)  # caso banale (m=1): nessun
                                                # sottoprocesso lanciato, nulla da misurare

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_dp_worker,
                        args=(p, dims, memory_limit_bytes, reconstruct_assignment,
                              force_representation, child_conn))
    proc.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout=None):
            (status, cmax, n_states, assignment, note, n_jobs_done, repr_used,
             solver_runtime, peak_memory_mb, peak_vsz_mb) = parent_conn.recv()
        else:
            (status, cmax, n_states, assignment, note, n_jobs_done, repr_used, solver_runtime,
             peak_memory_mb, peak_vsz_mb) = (
                "ERROR", None, -1, None, "nessuna risposta dal worker", None, None, None, None,
                None)
    except EOFError:
        (status, cmax, n_states, assignment, note, n_jobs_done, repr_used, solver_runtime,
         peak_memory_mb, peak_vsz_mb) = (
            "MEMORY_LIMIT_EXCEEDED", None, -1, None, None, None, None, None, None, None)
            
    proc.join()
    
    if note is None:
        # non e' arrivato nessun messaggio utile dal figlio (pipe chiusa
        # senza risposta): con il margine proattivo questo dovrebbe restare
        # raro, ma se succede l'exitcode e' l'unica informazione diagnostica
        # rimasta per distinguere un OOM del sistema (SIGKILL, exitcode
        # negativo) da un esaurimento avvenuto anche nel percorso di
        # recupero (exitcode 0, vedi commento in _dp_worker).
        note = f"pipe chiusa senza risposta dal sottoprocesso (exitcode={proc.exitcode})"

    if proc.exitcode not in (0, None) and status not in ("OPTIMAL", "MEMORY_LIMIT_EXCEEDED"):
        status, cmax, n_states, assignment = "MEMORY_LIMIT_EXCEEDED", None, -1, None
        note = f"sottoprocesso terminato con exitcode={proc.exitcode}"

    runtime = time.perf_counter() - start
    return DPResult(status=status, cmax=cmax, runtime=runtime,
                     n_states_final=n_states, assignment=assignment,
                     assignment_note=note,
                     n_jobs_dp_exact=(n if status == "OPTIMAL" else n_jobs_done),
                     representation=repr_used,
                     solver_runtime=solver_runtime,
                     peak_memory_mb=peak_memory_mb,
                     peak_vsz_mb=peak_vsz_mb)


if __name__ == "__main__":
    p = [90, 774, 655, 439, 434, 859, 86, 698, 202, 95]
    res = solve_dp(p, m=3)
    print(res)
    if res.assignment:
        loads = [0, 0, 0]
        for i, h in enumerate(res.assignment):
            loads[h] += p[i]
        print("Carichi:", loads, "max:", max(loads))

    # forzare esplicitamente l'una o l'altra rappresentazione (senza
    # ricostruzione, unico caso in cui la densa e' ammessa):
    for rep in ("sparse", "dense"):
        r = solve_dp(p, m=3, reconstruct_assignment=False, force_representation=rep)
        print(rep, "->", r.status, r.cmax, f"{r.runtime:.4f}s", r.representation)
