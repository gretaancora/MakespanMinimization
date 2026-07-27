"""
Confronto del tempo di calcolo dei tre metodi (MILP/Gurobi, DP, euristica LPT)
per ciascuna combinazione (m, n) -- pensato per i CSV prodotti da
experiment_runner.py (servono 'm', 'n', 'variance' piu' almeno una tra le
colonne di runtime dei tre metodi, vedi _METHOD_SPECS piu' sotto).

Per ogni combinazione (m, n) presente nei dati viene prodotto UN grafico a
barre distinto: sull'asse x le due categorie di varianza ("bassa varianza",
"alta varianza"), e per ciascuna un gruppo di barre, una per metodo
(MILP/DP/LPT). Se per una combinazione (m, n, variance, metodo) esistono piu'
istanze (tipico di experiment_runner.py con --instances-per-class > 1), la
barra mostra l'aggregato (mediana di default, vedi --agg).

SOLO IL TEMPO DEL SOLVER, non l'overhead attorno ad esso: per MILP e DP si
usano le colonne *_solver_runtime (rispettivamente Model.Runtime di Gurobi e
il tempo misurato DENTRO il sottoprocesso worker della DP, vedi
solvers/milp_solver.py e solvers/dp_solver.py), non *_runtime (che include
anche overhead di processo/wall-clock lato nostro: costruzione del modello
Gurobi fuori da Model.Runtime, fork()/join() del sottoprocesso DP, ecc.). Per
LPT si usa semplicemente lpt_runtime: l'euristica gira in-process senza
sottoprocessi ne' librerie esterne, quindi quel valore e' gia' puro tempo di
calcolo (vedi solvers/heuristic_solver.py).

Un metodo viene incluso nel grafico solo se la sua colonna di runtime e'
presente nel CSV fornito (utile ad es. quando il MILP non e' stato eseguito
perche' Gurobi non e' disponibile nell'ambiente: in quel caso si ottengono
comunque i grafici con le sole barre DP/LPT, invece di un errore).

Le barre che corrispondono a una combinazione (m, n, variance) in cui
ALMENO UNA istanza aggregata non ha chiuso in modo pulito -- MILP con
milp_status == "TIME_LIMIT" (tempo troncato al limite, non tempo di
risoluzione all'ottimo) o DP con dp_status == "MEMORY_LIMIT_EXCEEDED" (si
ferma quando esaurisce la memoria, vedi dp_solver.py) -- sono marcate con un
tratteggio e un bordo firebrick, stessa evidenziazione OOM usata in
plot_optimality_gap.py, cosi' da non leggere per errore un runtime troncato
come se fosse un tempo di risoluzione a convergenza. LPT non ha uno stato di questo tipo
(trova sempre una soluzione in O(n log n)), quindi non viene mai marcato.

Dato che il tratteggio scatta gia' con una sola istanza al limite (su
--instances-per-class istanze aggregate nella stessa barra), sopra ogni barra
tratteggiata compare anche un'etichetta 'k/n' (es. '1/10') col numero di
istanze effettivamente al limite su quelle aggregate: una minoranza (1/10,
l'altezza della barra resta un tempo di convergenza vero, dominato dai run
puliti) resta cosi' distinguibile a colpo d'occhio da un aggregato quasi o
del tutto troncato (9/10, 10/10, l'altezza e' essa stessa un valore
troncato), senza pero' nascondere il segnale che almeno un'istanza ha
comunque toccato il limite.

L'asse y e' in scala log per default (--linear-scale per disattivarla): i tre
metodi tipicamente differiscono di diversi ordini di grandezza (LPT: micro/
millisecondi: DP: da millisecondi a minuti; MILP: fino al time limit) sulla
stessa istanza, una scala lineare schiaccerebbe le barre piu' basse.

Uso da linea di comando:

    python analysis/plot_solver_time_by_variance.py \\
        --csv results/experiment1.csv results/experiment2.csv \\
        --out results/solver_time.png

produce un file per ciascuna combinazione (m, n) presente nei CSV, ad es.
results/solver_time_m2_n10.png, results/solver_time_m2_n25.png, ...

Oppure programmaticamente:

    from analysis.plot_solver_time_by_variance import (load_solver_time_results,
                                                          plot_solver_time_bars)
    df = load_solver_time_results(["results/experiment1.csv"])
    plot_solver_time_bars(df, out_path="results/solver_time.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance"}

# un metodo entra nel grafico solo se "runtime_col" e' presente nel CSV.
# "status_col"/"limit_status" sono opzionali: se presenti, le combinazioni
# (m, n, variance) in cui almeno un'istanza ha quello status vengono
# marcate con un tratteggio invece di una barra piena (vedi docstring).
_METHOD_SPECS = [
    ("MILP", "milp_solver_runtime", "milp_status", "TIME_LIMIT"),
    ("DP",   "dp_solver_runtime",   "dp_status",   "MEMORY_LIMIT_EXCEEDED"),
    ("LPT",  "lpt_runtime",         None,          None),
]

# palette fissa per metodo, cosi' lo stesso metodo ha sempre lo stesso colore
# anche confrontando grafici prodotti in run diverse (stesso principio delle
# palette fisse in plot_milp_scaling.py / plot_dp_calibration_bars.py)
_METHOD_COLORS = {"MILP": "tab:blue", "DP": "tab:orange", "LPT": "tab:green"}

# ordine fisso sull'asse x, come richiesto: bassa varianza poi alta varianza
_VARIANCE_ORDER = ["low", "high"]
_VARIANCE_LABELS = {"low": "bassa varianza", "high": "alta varianza"}

_LIMIT_HATCH = "//"
# stesso bordo usato per l'evidenziazione OOM in plot_optimality_gap.py
# (li' _OOM_EDGE), per rendere visivamente coerente la marcatura di
# time/memory limit tra i due script
_LIMIT_EDGE = "firebrick"
_BAR_GAP = 0.08  # spazio vuoto tra un gruppo di barre (variance) e il successivo


def load_solver_time_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Concatena i CSV indicati e rimuove eventuali righe duplicate (stesso
    instance_id) nel caso gli stessi punti compaiano in piu' file, stesso
    principio di load_milp_results / load_dp_calibration_results."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_solver_time_by_variance] rimosse {n_before - len(df)} righe "
                  f"duplicate (stesso instance_id in piu' file)")

    return df.sort_values(["m", "n", "variance"]).reset_index(drop=True)


def _available_methods(df: pd.DataFrame) -> List[tuple]:
    """Sottoinsieme di _METHOD_SPECS i cui runtime_col sono presenti nel
    CSV, nell'ordine fisso MILP/DP/LPT (non l'ordine di apparizione nel
    file, cosi' i colori/posizioni sono sempre consistenti tra grafici)."""
    return [spec for spec in _METHOD_SPECS if spec[1] in df.columns]


def plot_solver_time_bars(df: pd.DataFrame, m_values: Optional[List[int]] = None,
                           n_values: Optional[List[int]] = None, agg: str = "median",
                           log_scale: bool = True,
                           out_path: str = "results/solver_time.png") -> None:
    """Un file immagine per ciascuna combinazione (m, n) presente nei dati
    (filtrata da m_values/n_values se forniti). out_path e' usato come base:
    'results/solver_time.png' diventa 'results/solver_time_m2_n10.png',
    'results/solver_time_m2_n25.png', ecc. (stesso principio di out_path in
    plot_milp_scaling_by_variance / plot_dp_calibration_bars).

    agg: "median" (default) o "mean", usato per aggregare piu' istanze della
    stessa classe (m, n, variance, metodo) in un'unica altezza di barra."""
    if agg not in ("median", "mean"):
        raise ValueError(f"agg deve essere 'median' o 'mean' (ricevuto: {agg!r})")

    methods = _available_methods(df)
    if not methods:
        raise ValueError("nessuna colonna di runtime dei solver trovata nel CSV "
                          f"(attese una o piu' tra: {[spec[1] for spec in _METHOD_SPECS]})")

    sub = df
    if m_values is not None:
        sub = sub[sub["m"].isin(m_values)]
    if n_values is not None:
        sub = sub[sub["n"].isin(n_values)]
    if sub.empty:
        raise ValueError("nessuna riga da plottare dopo il filtro su m_values/n_values")

    root, ext = os.path.splitext(out_path)
    pairs = sub[["m", "n"]].drop_duplicates().sort_values(["m", "n"]).itertuples(index=False)
    for m, n in pairs:
        pair_sub = sub[(sub["m"] == m) & (sub["n"] == n)]
        pair_out = f"{root}_m{m}_n{n}{ext}"
        _plot_single_pair(pair_sub, m=m, n=n, methods=methods, agg=agg,
                           log_scale=log_scale, out_path=pair_out)


def _aggregate(rows: pd.DataFrame, runtime_col: str, agg: str) -> Optional[float]:
    values = rows[runtime_col].dropna()
    if values.empty:
        return None
    return float(values.median() if agg == "median" else values.mean())


def _limit_stats(rows: pd.DataFrame, runtime_col: str, status_col: Optional[str],
                  limit_status: Optional[str]) -> tuple:
    """Ritorna (hit, frac_label). Il tratteggio scatta ancora con il criterio
    'almeno un'istanza al limite' (una singola istanza troncata puo'
    comunque spostare la mediana/media in modo non trascurabile, quindi va
    comunque segnalata), MA frac_label riporta 'k/n' -- quante istanze, tra
    quelle con runtime_col non nullo (le stesse usate da _aggregate per
    calcolare l'altezza della barra), hanno effettivamente status_col ==
    limit_status. Cosi' una minoranza (es. 1/10, altezza = mediana di 9 run
    'puliti' + 1 troncato) resta visivamente distinguibile da un aggregato
    quasi/del tutto troncato (es. 10/10, l'altezza e' essa stessa un valore
    troncato), senza perdere il segnale che almeno un'istanza ha comunque
    toccato il limite (vedi analysis/plot_solver_time_by_n.py, stessa logica)."""
    if status_col is None or status_col not in rows.columns:
        return False, None
    valid = rows.dropna(subset=[runtime_col])
    total = len(valid)
    if total == 0:
        return False, None
    hit_count = int((valid[status_col] == limit_status).sum())
    if hit_count == 0:
        return False, None
    return True, f"{hit_count}/{total}"


def _plot_single_pair(sub: pd.DataFrame, m: int, n: int, methods: List[tuple], agg: str,
                       log_scale: bool, out_path: str) -> None:
    present_variances = [v for v in _VARIANCE_ORDER if v in sub["variance"].unique()]
    if not present_variances:
        print(f"[plot_solver_time_by_variance] nessuna varianza nota per m={m}, n={n}, salto")
        return

    n_methods = len(methods)
    bar_width = (1.0 - _BAR_GAP) / n_methods
    group_positions = {v: i for i, v in enumerate(present_variances)}
    # offset del centro di ciascuna barra all'interno del gruppo, simmetrico
    # attorno alla posizione del gruppo (stesso schema di offsets in
    # plot_dp_calibration_bars.py, generalizzato a n_methods invece di 2)
    offsets = {name: (j - (n_methods - 1) / 2) * bar_width for j, (name, *_ ) in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(present_variances) * n_methods), 6.0))

    limit_hit_any = False
    for method_name, runtime_col, status_col, limit_status in methods:
        positions, heights = [], []
        colors, hatches, frac_labels = [], [], []
        for variance in present_variances:
            rows = sub[sub["variance"] == variance]
            height = _aggregate(rows, runtime_col, agg)
            if height is None:
                continue  # niente barra "vuota" se manca la misurazione (vedi docstring)
            positions.append(group_positions[variance] + offsets[method_name])
            heights.append(height)
            hit, frac_label = _limit_stats(rows, runtime_col, status_col, limit_status)
            colors.append(_METHOD_COLORS[method_name])
            hatches.append(_LIMIT_HATCH if hit else None)
            frac_labels.append(frac_label)
            limit_hit_any = limit_hit_any or hit

        if not positions:
            continue

        for pos, height, color, hatch, frac_label in zip(positions, heights, colors, hatches, frac_labels):
            ax.bar(pos, height, width=bar_width * 0.92, color=color,
                   edgecolor=_LIMIT_EDGE if hatch else "none", hatch=hatch,
                   linewidth=1.3 if hatch else 0.0, label="_nolegend_")
            if frac_label is not None:
                # etichetta 'k/n' (istanze al limite su istanze aggregate),
                # stesso principio di plot_solver_time_by_n.py: distingue una
                # minoranza (es. 1/10) da un aggregato quasi/del tutto
                # troncato (es. 10/10) pur restando entrambi tratteggiati
                y_text = height * 1.08 if log_scale else height
                ax.text(pos, y_text, frac_label, ha="center", va="bottom",
                        fontsize=7.5, color=_LIMIT_EDGE, clip_on=False)

    if log_scale:
        ax.set_yscale("log")

    if limit_hit_any:
        # spazio extra in alto per non tagliare le etichette 'k/n' appena
        # aggiunte (l'autoscale di matplotlib considera le barre ma non il
        # testo). IMPORTANTE: questo va fatto DOPO aver impostato la scala
        # log, non prima -- get_ylim()/set_ylim() devono leggere/scrivere i
        # limiti gia' in scala log (dove il minimo non puo' essere 0),
        # altrimenti si riparte da limiti calcolati in scala lineare (dove
        # il minimo di un bar chart e' 0) e passare alla scala log DOPO
        # rompe tutto l'asse (barre e ytick fuori posto).
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 3 if log_scale else top + 0.12 * (top - bottom))

    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels([_VARIANCE_LABELS[v] for v in present_variances])
    ax.set_xlabel("varianza dei tempi di lavorazione")
    ax.set_ylabel(f"tempo di calcolo del solver ({agg}, s{', scala log' if log_scale else ''})")
    ax.set_title(f"Tempo di calcolo dei solver (m={m}, n={n})")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = [Patch(facecolor=_METHOD_COLORS[name], label=name) for name, *_ in methods]
    if limit_hit_any:
        handles.append(Patch(facecolor="white", edgecolor=_LIMIT_EDGE, hatch=_LIMIT_HATCH,
                              label="time/memory limit raggiunto"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_solver_time_by_variance] grafico salvato in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati (experiment_runner.py) da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=None,
                         help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--n", nargs="+", type=int, default=None,
                         help="valori di n da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra piu' istanze della stessa classe "
                              "(m, n, variance, metodo) (default: median)")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false", default=True,
                         help="asse y in scala lineare invece che log (default: log)")
    parser.add_argument("--out", default="results/solver_time.png",
                         help="path base del file immagine di output: un file per ciascuna "
                              "combinazione (m, n), vedi docstring del modulo")
    args = parser.parse_args()

    df = load_solver_time_results(args.csv)
    plot_solver_time_bars(df, m_values=args.m, n_values=args.n, agg=args.agg,
                           log_scale=args.log_scale, out_path=args.out)


if __name__ == "__main__":
    main()
