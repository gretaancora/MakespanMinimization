"""
Confronto del tempo di calcolo dei tre metodi (MILP/Gurobi, DP, euristica LPT)
al VARIARE DI n -- pensato per gli stessi CSV prodotti da experiment_runner.py
usati da plot_solver_time_by_variance.py (servono 'm', 'n', 'variance' piu'
almeno una tra le colonne di runtime dei tre metodi, vedi _METHOD_SPECS piu'
sotto). E' il "trasposto" di plot_solver_time_by_variance.py: li' l'asse x
erano le due varianze (un grafico per (m, n)), qui l'asse x e' n (un grafico
per (m, variance)).

Per ogni combinazione (m, variance) presente nei dati viene prodotto UN
grafico a barre distinto: sull'asse x i valori di n presenti per quella
combinazione, e per ciascuno un gruppo di barre, una per metodo (MILP/DP/
LPT). Se per una combinazione (m, n, variance, metodo) esistono piu' istanze
(tipico di experiment_runner.py con --instances-per-class > 1), la barra
mostra l'aggregato (mediana di default, vedi --agg) -- stesso principio di
plot_solver_time_by_variance.py.

SOLO IL TEMPO DEL SOLVER, non l'overhead attorno ad esso: per MILP e DP si
usano le colonne *_solver_runtime (rispettivamente Model.Runtime di Gurobi e
il tempo misurato DENTRO il sottoprocesso worker della DP, vedi
solvers/milp_solver.py e solvers/dp_solver.py), non *_runtime (che include
anche overhead di processo/wall-clock lato nostro). Per LPT si usa
semplicemente lpt_runtime (nessun sottoprocesso, gia' puro tempo di calcolo,
vedi solvers/heuristic_solver.py).

Un metodo viene incluso nel grafico solo se la sua colonna di runtime e'
presente nel CSV fornito (utile ad es. quando il MILP non e' stato eseguito
perche' Gurobi non e' disponibile nell'ambiente).

Le barre che corrispondono a una combinazione (m, n, variance) in cui ALMENO
UNA istanza aggregata non ha chiuso in modo pulito -- MILP con milp_status ==
"TIME_LIMIT" o DP con dp_status == "MEMORY_LIMIT_EXCEEDED" -- sono marcate
con un tratteggio e un bordo firebrick, stessa evidenziazione OOM/time-limit
usata in plot_optimality_gap.py e in plot_solver_time_by_variance.py, cosi'
da non leggere per errore un runtime troncato come se fosse un tempo di
risoluzione a convergenza. LPT non ha uno stato di questo tipo, quindi non
viene mai marcato.

Dato che il tratteggio scatta gia' con una sola istanza al limite (su
--instances-per-class istanze aggregate nella stessa barra), sopra ogni
barra tratteggiata compare anche un'etichetta 'k/n' (es. '1/10') col numero
di istanze effettivamente al limite su quelle aggregate: una minoranza (1/10,
l'altezza della barra resta un tempo di convergenza vero, dominato dai run
puliti) resta cosi' distinguibile a colpo d'occhio da un aggregato quasi o
del tutto troncato (9/10, 10/10, l'altezza e' essa stessa un valore
troncato), senza pero' nascondere il segnale che almeno un'istanza ha
comunque toccato il limite.

L'asse y e' in scala log per default (--linear-scale per disattivarla),
stesso motivo di plot_solver_time_by_variance.py: i tre metodi tipicamente
differiscono di diversi ordini di grandezza sulla stessa istanza.

Uso da linea di comando:

    python analysis/plot_solver_time_by_n.py \\
        --csv results/experiment1.csv results/experiment2.csv \\
        --out results/solver_time_by_n.png

produce un file per ciascuna combinazione (m, variance) presente nei CSV, ad
es. results/solver_time_by_n_m2_low.png, results/solver_time_by_n_m2_high.png, ...

Oppure programmaticamente:

    from analysis.plot_solver_time_by_n import (load_solver_time_results,
                                                  plot_solver_time_by_n_bars)
    df = load_solver_time_results(["results/experiment1.csv"])
    plot_solver_time_by_n_bars(df, out_path="results/solver_time_by_n.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance"}

# stessi _METHOD_SPECS/_METHOD_COLORS di plot_solver_time_by_variance.py: un
# metodo entra nel grafico solo se "runtime_col" e' presente nel CSV, e la
# stessa palette fissa per metodo garantisce colori consistenti tra i due
# script (e tra run diverse dello stesso script).
_METHOD_SPECS = [
    ("MILP", "milp_solver_runtime", "milp_status", "TIME_LIMIT"),
    ("DP",   "dp_solver_runtime",   "dp_status",   "MEMORY_LIMIT_EXCEEDED"),
    ("LPT",  "lpt_runtime",         None,          None),
]
_METHOD_COLORS = {"MILP": "tab:blue", "DP": "tab:orange", "LPT": "tab:green"}

# ordine fisso delle varianze per l'iterazione (un grafico per ciascuna,
# non un asse), stessa etichettatura in italiano di plot_solver_time_by_variance.py
_VARIANCE_ORDER = ["low", "high"]
_VARIANCE_LABELS = {"low": "bassa varianza", "high": "alta varianza"}

_LIMIT_HATCH = "//"
# stesso bordo usato per l'evidenziazione OOM/time-limit in
# plot_optimality_gap.py e plot_solver_time_by_variance.py
_LIMIT_EDGE = "firebrick"
_BAR_GAP = 0.08  # spazio vuoto tra un gruppo di barre (valore di n) e il successivo


def load_solver_time_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Identica a load_solver_time_results di plot_solver_time_by_variance.py:
    concatena i CSV indicati e rimuove eventuali righe duplicate (stesso
    instance_id) nel caso gli stessi punti compaiano in piu' file."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_solver_time_by_n] rimosse {n_before - len(df)} righe "
                  f"duplicate (stesso instance_id in piu' file)")

    return df.sort_values(["m", "variance", "n"]).reset_index(drop=True)


def _available_methods(df: pd.DataFrame) -> List[tuple]:
    """Sottoinsieme di _METHOD_SPECS i cui runtime_col sono presenti nel
    CSV, nell'ordine fisso MILP/DP/LPT."""
    return [spec for spec in _METHOD_SPECS if spec[1] in df.columns]


def plot_solver_time_by_n_bars(df: pd.DataFrame, m_values: Optional[List[int]] = None,
                                variance_values: Optional[List[str]] = None,
                                agg: str = "median", log_scale: bool = True,
                                out_path: str = "results/solver_time_by_n.png") -> None:
    """Un file immagine per ciascuna combinazione (m, variance) presente nei
    dati (filtrata da m_values/variance_values se forniti). out_path e' usato
    come base: 'results/solver_time_by_n.png' diventa
    'results/solver_time_by_n_m2_low.png', 'results/solver_time_by_n_m2_high.png',
    ecc. (stesso principio di out_path in plot_solver_time_by_variance.py).

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
    if variance_values is not None:
        sub = sub[sub["variance"].isin(variance_values)]
    if sub.empty:
        raise ValueError("nessuna riga da plottare dopo il filtro su m_values/variance_values")

    root, ext = os.path.splitext(out_path)
    pairs = sub[["m", "variance"]].drop_duplicates()
    # ordine fisso: m crescente, poi low/high (non l'ordine di apparizione nel file)
    pairs["_variance_order"] = pairs["variance"].map({v: i for i, v in enumerate(_VARIANCE_ORDER)})
    pairs = pairs.sort_values(["m", "_variance_order"]).itertuples(index=False)
    for m, variance, _ in pairs:
        pair_sub = sub[(sub["m"] == m) & (sub["variance"] == variance)]
        pair_out = f"{root}_m{m}_{variance}{ext}"
        _plot_single_pair(pair_sub, m=m, variance=variance, methods=methods, agg=agg,
                           log_scale=log_scale, out_path=pair_out)


def _aggregate(rows: pd.DataFrame, runtime_col: str, agg: str) -> Optional[float]:
    values = rows[runtime_col].dropna()
    if values.empty:
        return None
    return float(values.median() if agg == "median" else values.mean())


def _limit_stats(rows: pd.DataFrame, runtime_col: str, status_col: Optional[str],
                  limit_status: Optional[str]) -> tuple:
    """Ritorna (hit, frac_label). Il tratteggio scatta ancora con il criterio
    'almeno un'istanza al limite' (stesso criterio, invariato, di
    plot_solver_time_by_variance.py: una singola istanza troncata puo'
    comunque spostare la mediana/media in modo non trascurabile, quindi va
    comunque segnalata), MA frac_label riporta 'k/n' -- quante istanze, tra
    quelle con runtime_col non nullo (le stesse usate da _aggregate per
    calcolare l'altezza della barra), hanno effettivamente status_col ==
    limit_status. Cosi' una minoranza (es. 1/10, altezza = mediana di 9 run
    'puliti' + 1 troncato) resta visivamente distinguibile da un aggregato
    quasi/del tutto troncato (es. 10/10), senza perdere il segnale che
    ALMENO un'istanza ha comunque toccato il limite."""
    if status_col is None or status_col not in rows.columns:
        return False, None
    valid = rows.dropna(subset=[runtime_col])
    total = len(valid)
    if total == 0:
        return False, None
    hit_count = int((valid[status_col] == limit_status).sum())
    if hit_count == 0:
        return False, None
    return True, f"OOM({hit_count}/{total})"


def _plot_single_pair(sub: pd.DataFrame, m: int, variance: str, methods: List[tuple], agg: str,
                       log_scale: bool, out_path: str) -> None:
    present_n = sorted(sub["n"].unique())
    if not present_n:
        print(f"[plot_solver_time_by_n] nessun valore di n per m={m}, variance={variance}, salto")
        return

    n_methods = len(methods)
    bar_width = (1.0 - _BAR_GAP) / n_methods
    group_positions = {n: i for i, n in enumerate(present_n)}
    # offset del centro di ciascuna barra all'interno del gruppo, simmetrico
    # attorno alla posizione del gruppo (stesso schema di
    # plot_solver_time_by_variance.py, qui i gruppi sono i valori di n
    # invece delle due varianze)
    offsets = {name: (j - (n_methods - 1) / 2) * bar_width for j, (name, *_ ) in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(present_n) * n_methods), 6.0))

    limit_hit_any = False
    for method_name, runtime_col, status_col, limit_status in methods:
        positions, heights = [], []
        colors, hatches, frac_labels = [], [], []
        for n in present_n:
            rows = sub[sub["n"] == n]
            height = _aggregate(rows, runtime_col, agg)
            if height is None:
                continue  # niente barra "vuota" se manca la misurazione (vedi docstring)
            positions.append(group_positions[n] + offsets[method_name])
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
                # etichetta 'k/n' (istanze al limite su istanze aggregate)
                # appena sopra la barra, stesso colore del bordo di
                # evidenziazione, cosi' una minoranza (es. 1/10) resta
                # distinguibile a colpo d'occhio da un aggregato quasi/del
                # tutto troncato (es. 10/10) pur restando entrambi tratteggiati
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
    ax.set_xticklabels([str(n) for n in present_n])
    ax.set_xlabel("numero di job (n)")
    ax.set_ylabel(f"tempo di calcolo del solver ({agg}, s{', scala log' if log_scale else ''})")
    ax.set_title(f"Tempo di calcolo dei solver al variare di n "
                 f"(m={m}, {_VARIANCE_LABELS[variance]})")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = [Patch(facecolor=_METHOD_COLORS[name], label=name) for name, *_ in methods]
    if limit_hit_any:
        handles.append(Patch(facecolor="white", edgecolor=_LIMIT_EDGE, hatch=_LIMIT_HATCH,
                              label="time/memory limit raggiunto"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_solver_time_by_n] grafico salvato in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati (experiment_runner.py) da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=None,
                         help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--variance", nargs="+", choices=["low", "high"], default=None,
                         help="valori di variance da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra piu' istanze della stessa classe "
                              "(m, n, variance, metodo) (default: median)")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false", default=True,
                         help="asse y in scala lineare invece che log (default: log)")
    parser.add_argument("--out", default="results/solver_time_by_n.png",
                         help="path base del file immagine di output: un file per ciascuna "
                              "combinazione (m, variance), vedi docstring del modulo")
    args = parser.parse_args()

    df = load_solver_time_results(args.csv)
    plot_solver_time_by_n_bars(df, m_values=args.m, variance_values=args.variance, agg=args.agg,
                               log_scale=args.log_scale, out_path=args.out)


if __name__ == "__main__":
    main()
