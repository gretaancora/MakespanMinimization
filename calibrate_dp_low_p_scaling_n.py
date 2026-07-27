"""
Calibrazione DP: tempi di processamento "pochi unita'" (come in
calibrate_dp_low_p_many_m.py), ma questa volta a m FISSO e n crescente --
l'asse complementare rispetto a quello script.

Motivazione
-----------
calibrate_dp_low_p_many_m.py tiene il Cmax atteso in una fascia stretta e
fissa (--target-cmax) e fa crescere m, per isolare l'effetto della
componente esponenziale in dims = m-1 dalla dimensione P = sum(p_i) del
problema. Qui la domanda e' l'altra meta' della stessa storia: A PARITA'
DI m, quanto puo' crescere n (e quindi P, visto che i tempi restano
piccoli e fissi in [--p-low, --p-high]) prima che la DP smetta di
chiudere OPTIMAL entro il budget di tempo/memoria dato? Qui NON si tiene
fisso il Cmax atteso (--target-cmax non esiste in questo script,
deliberatamente).

Per ciascun valore di m nella griglia (--m-values), n parte da
max(--n-start, m+1) e cresce moltiplicativamente (fattore --growth-
factor, default 2.0, con un incremento minimo di +1) finche', per un
dato n, NESSUNA delle --instances-per-class istanze (seed diversi) chiude
OPTIMAL entro il soft cap di tempo e il budget di memoria dati -- a quel
punto si smette di crescere per QUEL m (o per QUELLA rappresentazione, se
--representation both, vedi sotto) e si passa al successivo. Finche'
ALMENO UNA istanza chiude OPTIMAL a un dato n, si continua a crescere:
questo lascia visibile nel CSV anche la eventuale zona di transizione
(alcuni seed falliscono, altri no, allo stesso n) invece di fermarsi al
primo fallimento isolato.

Un --n-max (default 2_000_000) fa da rete di sicurezza per evitare un
loop infinito nel caso patologico in cui la DP non fallisca mai entro
quella soglia.

Rappresentazione (--representation, default "auto")
-----------------------------------------------------
  - "auto" (default): force_representation=None passato a solve_dp --
    per ciascuna istanza viene usata la scelta automatica di
    choose_representation/_use_dense (vedi dp_solver.py), quella che si
    avrebbe davvero in campagna. La colonna 'representation' nel CSV
    riporta dp.representation, cioe' cosa e' stato REALMENTE usato per
    quella riga (puo' cambiare da un n all'altro dentro lo stesso m, se
    P supera la soglia che fa preferire la sparsa alla densa via
    _use_dense) -- non e' detto coincida sempre con 'auto_pick' (che e'
    calcolato a parte con choose_representation prima della chiamata,
    solo a scopo di confronto/tracciabilita').
  - "sparse" / "dense": forza quella rappresentazione con
    force_representation, per OGNI istanza indipendentemente da quale
    avrebbe scelto choose_representation -- utile per scoprire il limite
    pratico reale di una rappresentazione specifica invece di quello solo
    stimato (stessa logica di calibrate_dp_lpt.py). La colonna
    'representation' del CSV coincide sempre col valore forzato (salvo il
    caso banale m=1, qui escluso dalla griglia di default).
  - "both": esegue e confronta ENTRAMBE le rappresentazioni sulla STESSA
    istanza per ciascun (m, n, seed), come calibrate_dp_lpt.py, ma con la
    crescita dinamica di n invece di una griglia fissa: ciascuna
    rappresentazione cresce in n E SI FERMA IN MODO INDIPENDENTE (una
    puo' continuare a salire mentre l'altra si e' gia' fermata) -- si
    passa al prossimo m solo quando ENTRAMBE si sono fermate. Raddoppia
    il numero di solve per istanza rispetto agli altri tre modi.

dp_status nel CSV distingue gli stessi tre casi di calibrate_dp_lpt.py:
  - OPTIMAL: la DP ha chiuso entro budget di tempo/memoria;
  - MEMORY_LIMIT_EXCEEDED: memoria esaurita davvero (dp_cmax, quando
    presente, e' l'incumbent parziale sui soli job trattati prima del
    crash -- NON un cmax ammissibile per l'istanza intera, vedi
    dp_solver.py e la discussione in plot_dp_low_p_many_m.py);
  - TIME_SOFT_CAP_STOPPED (in stop_reason, dp_status resta OPTIMAL): la
    DP e' arrivata comunque all'ottimo, solo oltre il tempo soft cap --
    NON e' un fallimento della DP, e' una scelta di QUESTO script.

Ogni riga del CSV di dettaglio (--out) e' una singola istanza (m, n,
seed, representation): riporta dp_peak_memory_mb (RSS) e dp_peak_vsz_mb
(VmPeak, virtuale -- la metrica su cui agisce davvero il budget
--memory-limit-gb via RLIMIT_AS, vedi _peak_vsz_mb in dp_solver.py), e il
Cmax dell'istanza in due varianti: dp_cmax (ammissibile solo se
dp_status == "OPTIMAL") e reference_cmax (LPT, sempre ammissibile). Il
CSV di riepilogo (--summary-out) aggrega per m (e per representation, se
--representation both): fino a che n la DP ha chiuso OPTIMAL su almeno
una istanza, e a partire da che n fallisce su tutte.

Uso:
    python3 calibrate_dp_low_p_scaling_n.py [--p-low 1] [--p-high 5]
        [--m-values 3 4 5 6] [--n-start 10] [--growth-factor 2.0]
        [--representation auto|sparse|dense|both]
        [--instances-per-class 3] [--memory-limit-gb 16]
        [--time-soft-cap-s 120]
"""
import argparse
import sys, os, csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from instances.instance_generator import generate_instance_custom_range
from solvers.dp_solver import solve_dp, choose_representation
from solvers.heuristic_solver import solve_lpt

DEFAULT_M_VALUES = [3, 4, 5, 6]


def _next_n(n: int, growth_factor: float) -> int:
    return max(n + 1, round(n * growth_factor))


def _active_representations(mode: str):
    # ritorna la lista di rappresentazioni "forzate" da eseguire nel loop
    # (usata anche come chiave del dict 'stopped'); per "auto" c'e' un solo
    # elemento (force_representation=None passato a solve_dp), etichettato
    # "auto" solo per il bookkeeping del loop -- la rappresentazione REALE
    # usata riga per riga viene letta da dp.representation, non da qui.
    if mode == "both":
        return ["sparse", "dense"]
    return [mode]  # "auto", "sparse" oppure "dense"


def _build_summary(rows, representation_mode):
    """Aggrega per m (e per representation se representation_mode=='both',
    dato che in quel caso sparse/dense sono due traiettorie di crescita
    indipendenti e non ha senso mediarle insieme)."""
    split_by_repr = representation_mode == "both"
    by_group = defaultdict(list)
    for r in rows:
        key = (r["m"], r["representation"]) if split_by_repr else (r["m"],)
        by_group[key].append(r)

    summary = []
    for key in sorted(by_group.keys()):
        rs = by_group[key]
        m = key[0]
        by_n = defaultdict(list)
        for r in rs:
            by_n[r["n"]].append(r)
        ns_sorted = sorted(by_n.keys())

        last_n_some_optimal = None
        first_n_all_failed = None
        for n in ns_sorted:
            rs_n = by_n[n]
            n_optimal = sum(1 for r in rs_n if r["dp_status"] == "OPTIMAL" and r["stop_reason"] is None)
            if n_optimal > 0:
                last_n_some_optimal = n
            elif first_n_all_failed is None:
                first_n_all_failed = n

        if first_n_all_failed is None:
            recap = f"non ancora fallita entro n={ns_sorted[-1]} (limite raggiunto nella crescita)"
        elif last_n_some_optimal is None:
            recap = f"mai chiusa OPTIMAL, fallita gia' al primo n testato (n={ns_sorted[0]})"
        else:
            recap = (f"OK fino ad almeno n={last_n_some_optimal}, fallisce sistematicamente "
                     f"a partire da n={first_n_all_failed}")

        entry = dict(m=m, dims=m - 1)
        if split_by_repr:
            entry["representation"] = key[1]
        else:
            reps_seen = sorted({r["representation"] for r in rs})
            entry["representation"] = (representation_mode if len(reps_seen) <= 1 or representation_mode != "auto"
                                        else "auto (" + "->".join(reps_seen) + ")")
        entry.update(n_tested_min=ns_sorted[0], n_tested_max=ns_sorted[-1],
                     last_n_some_optimal=last_n_some_optimal,
                     first_n_all_failed=first_n_all_failed, recap=recap)
        summary.append(entry)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-low", type=int, default=1, help="estremo inferiore range tempi (incluso)")
    ap.add_argument("--p-high", type=int, default=5, help="estremo superiore range tempi (incluso)")
    ap.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
                     help="griglia di numero di macchine da testare (m fisso per ciascun test)")
    ap.add_argument("--n-start", type=int, default=10,
                     help="n di partenza per la crescita (viene comunque alzato ad almeno m+1)")
    ap.add_argument("--growth-factor", type=float, default=2.0,
                     help="fattore moltiplicativo di crescita di n a ogni passo (default 2.0)")
    ap.add_argument("--n-max", type=int, default=2_000_000,
                     help="rete di sicurezza, vedi docstring del modulo")
    ap.add_argument("--representation", choices=["auto", "sparse", "dense", "both"], default="auto",
                     help="'auto' (default): scelta automatica di solve_dp, come si avrebbe "
                          "davvero in campagna. 'sparse'/'dense': forza quella rappresentazione "
                          "per ogni istanza. 'both': esegue e confronta entrambe sulla stessa "
                          "istanza, con crescita di n indipendente per ciascuna -- vedi docstring")
    ap.add_argument("--instances-per-class", type=int, default=3,
                     help="quante istanze (seed diversi) per ciascuna combinazione (m, n)")
    ap.add_argument("--memory-limit-gb", type=float, default=16.0,
                     help="deve corrispondere al budget usato nella campagna finale")
    ap.add_argument("--time-soft-cap-s", type=float, default=120.0,
                     help="oltre questo tempo per una singola istanza, si segnala "
                          "TIME_SOFT_CAP_STOPPED e l'istanza conta come fallita ai fini della "
                          "crescita di n")
    ap.add_argument("--out", type=str, default="results/calibration_dp_low_p_scaling_n.csv")
    ap.add_argument("--summary-out", type=str,
                     default="results/calibration_dp_low_p_scaling_n_summary.csv")
    ap.add_argument("--base-seed", type=int, default=9000,
                     help="seed di partenza (offset per non collidere con gli altri script di "
                          "calibrazione)")
    args = ap.parse_args()

    if args.p_low < 1 or args.p_high < args.p_low:
        ap.error(f"range non valido: --p-low {args.p_low} --p-high {args.p_high}")
    if args.growth_factor <= 1.0:
        ap.error(f"--growth-factor deve essere > 1.0: {args.growth_factor}")

    memory_limit_bytes = int(args.memory_limit_gb * 1024**3)
    active_reps = _active_representations(args.representation)  # chiavi del loop/stopped-dict

    rows = []
    print(f"p in [{args.p_low},{args.p_high}]  growth_factor={args.growth_factor}  "
          f"representation={args.representation}  memory_limit={args.memory_limit_gb}GB  "
          f"time_soft_cap={args.time_soft_cap_s}s\n")
    print(f"{'m':>3s} {'dims':>4s} {'n':>7s} {'seed':>6s} {'rep':>6s} {'auto':>6s}  "
          f"{'dp_status':>22s} {'dp_runtime':>10s} {'dp_rss_mb':>10s} {'dp_vsz_mb':>10s} "
          f"{'dp_states':>12s} {'lpt_cmax':>8s} {'dp_cmax':>8s}")

    for m in args.m_values:
        n = max(args.n_start, m + 1)
        stopped = {rep: False for rep in active_reps}
        while n <= args.n_max:
            n_optimal_this_n = {rep: 0 for rep in active_reps}
            for k in range(args.instances_per_class):
                seed = args.base_seed + 1_000_000 * m + 1_000 * n + k
                inst = generate_instance_custom_range(n=n, m=m, seed=seed,
                                                        p_low=args.p_low, p_high=args.p_high)
                lpt = solve_lpt(inst.processing_times, m)
                auto_pick = choose_representation(inst.P, m - 1, memory_limit_bytes)

                for rep in active_reps:
                    if stopped[rep]:
                        continue
                    force = None if rep == "auto" else rep
                    dp = solve_dp(inst.processing_times, m, memory_limit_bytes=memory_limit_bytes,
                                   reconstruct_assignment=False, force_representation=force)

                    stop_reason = None
                    if dp.status == "OPTIMAL" and dp.runtime > args.time_soft_cap_s:
                        stop_reason = "TIME_SOFT_CAP_STOPPED"
                    elif dp.status == "MEMORY_LIMIT_EXCEEDED":
                        stop_reason = "MEMORY_LIMIT_EXCEEDED"
                    elif dp.status == "ERROR":
                        stop_reason = "ERROR"
                    if dp.status == "OPTIMAL" and stop_reason is None:
                        n_optimal_this_n[rep] += 1

                    mem_str = f"{dp.peak_memory_mb:.1f}" if dp.peak_memory_mb is not None else "n/a"
                    vsz_str = f"{dp.peak_vsz_mb:.1f}" if dp.peak_vsz_mb is not None else "n/a"
                    dp_cmax_str = str(dp.cmax) if dp.cmax is not None else "n/a"
                    print(f"{m:3d} {m - 1:4d} {n:7d} {seed:6d} {dp.representation or rep:>6s} "
                          f"{auto_pick:>6s}  {dp.status:>22s} {dp.runtime:10.3f} {mem_str:>10s} "
                          f"{vsz_str:>10s} {dp.n_states_final:12d} {lpt.cmax:8d} {dp_cmax_str:>8s}"
                          + (f"   <- {stop_reason}" if stop_reason else ""))

                    rows.append(dict(m=m, n=n, seed=seed, p_low=args.p_low, p_high=args.p_high,
                                      representation=dp.representation or rep, auto_pick=auto_pick,
                                      dp_status=dp.status, dp_cmax=dp.cmax,
                                      reference_cmax=lpt.cmax,
                                      dp_runtime=round(dp.runtime, 3),
                                      dp_peak_memory_mb=(round(dp.peak_memory_mb, 1)
                                                          if dp.peak_memory_mb is not None else None),
                                      dp_peak_vsz_mb=(round(dp.peak_vsz_mb, 1)
                                                      if dp.peak_vsz_mb is not None else None),
                                      dp_states=dp.n_states_final,
                                      lpt_runtime=round(lpt.runtime, 5),
                                      stop_reason=stop_reason))

            newly_stopped = []
            for rep in active_reps:
                if not stopped[rep] and n_optimal_this_n[rep] == 0:
                    stopped[rep] = True
                    newly_stopped.append(rep)
            if newly_stopped:
                print(f"  (m={m}: '{', '.join(newly_stopped)}' non ha chiuso OPTIMAL nessuna istanza "
                      f"a n={n}, si ferma qui)")
            if all(stopped.values()):
                break
            n = _next_n(n, args.growth_factor)
        else:
            print(f"  (m={m}: raggiunto --n-max={args.n_max} senza che tutte le rappresentazioni "
                  f"attive fallissero, mi fermo per rete di sicurezza)")

    if not rows:
        print("\nnessuna riga prodotta, controlla i parametri (--m-values vuoto?)")
        return

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDettaglio salvato in {out_path} ({len(rows)} righe)")

    summary = _build_summary(rows, args.representation)
    summary_path = os.path.join(os.path.dirname(__file__), args.summary_out)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Riepilogo salvato in {summary_path}\n")

    print(f"{'m':>3s} {'dims':>4s} {'rep':>14s} {'n testati':>12s} {'ultimo n ok':>12s} "
          f"{'primo n KO':>11s}  recap")
    for s in summary:
        print(f"{s['m']:3d} {s['dims']:4d} {s['representation']:>14s} "
              f"{str(s['n_tested_min']) + '-' + str(s['n_tested_max']):>12s} "
              f"{str(s['last_n_some_optimal']):>12s} {str(s['first_n_all_failed']):>11s}  {s['recap']}")


if __name__ == "__main__":
    main()
