"""
Confronto del tempo di calcolo dei tre metodi (MILP/Gurobi, DP, euristica LPT)
al VARIARE DI n -- stessa identica logica di plot_solver_time_by_n.py, con
UNA differenza: quello script assume 'variance' in {'low', 'high'} (le due
classi fisse di experiment_runner.py), qui invece i CSV vengono da
experiments/run_low_p_experiment.py, dove 'variance' e' la stringa
descrittiva prodotta da generate_instance_custom_range() -- tipicamente
"custom[p_low,p_high]" (es. "custom[1,5]"), diversa run per run a seconda
di --p-low/--p-high, quindi non hardcodabile in una lista fissa come
_VARIANCE_ORDER/_VARIANCE_LABELS nello script gemello. Le uniche differenze
rispetto a plot_solver_time_by_n.py sono percio' _variance_sort_key() e
_variance_label() piu' sotto (niente scelte fisse "low"/"high" nel parsing
degli argomenti): il resto -- aggregazione, evidenziazione OOM/time-limit
con etichetta 'k/n', scala log, un file per (m, variance) -- e' identico,
vedi quel modulo per i dettagli.

_variance_label() riconosce comunque il pattern "custom[a,b]" e lo rende
leggibile come "p in [a,b]" invece di stampare la stringa grezza; valori
"low"/"high" restano etichettati come nello script gemello (bassa/alta
varianza), cosi' questo script funziona anche su CSV misti o sulla campagna
classica, non solo su quella a p basso.

Uso da linea di comando:

    python analysis/plot_solver_time_by_n_low_p.py \\
        --csv results/results_low_p.csv \\
        --out results/solver_time_by_n_low_p.png

produce un file per ciascuna combinazione (m, variance) presente nel CSV,
ad es. results/solver_time_by_n_low_p_m3_p1-5.png, ...
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance"}

# identici a plot_solver_time_by_n.py: un metodo entra nel grafico solo se
# "runtime_col" e' presente nel CSV, stessa palette fissa per metodo
_METHOD_SPECS = [
    ("MILP", "milp_solver_runtime", "milp_status", "TIME_LIMIT"),
    ("DP",   "dp_solver_runtime",   "dp_status",   "MEMORY_LIMIT_EXCEEDED"),
    ("LPT",  "lpt_runtime",         None,          None),
]
_METHOD_COLORS = {"MILP": "tab:blue", "DP": "tab:orange", "LPT": "tab:green"}

_LIMIT_HATCH = "//"
_LIMIT_EDGE = "firebrick"
_BAR_GAP = 0.08

_CUSTOM_RANGE_RE = re.compile(r"^custom\[(\d+),\s*(\d+)\]$")
# valori noti (dalla campagna low/high di experiment_runner.py) ordinati
# esplicitamente; qualunque altra stringa di variance (es. "custom[1,5]")
# finisce dopo, in ordine alfabetico tra loro -- non richiede di sapere in
# anticipo quali range custom compariranno nel CSV
_KNOWN_VARIANCE_ORDER = {"low": 0, "high": 1}


def _variance_sort_key(variance: str):
    return (_KNOWN_VARIANCE_ORDER.get(variance, 2), str(variance))


def _variance_label(variance: str) -> str:
    m = _CUSTOM_RANGE_RE.match(str(variance))
    if m:
        return f"p in [{m.group(1)},{m.group(2)}]"
    return {"low": "bassa varianza", "high": "alta varianza"}.get(variance, str(variance))


def _variance_slug(variance: str) -> str:
    """Per il nome file: 'custom[1,5]' -> 'p1-5' invece della stringa grezza
    (parentesi quadre e virgola sono scomode da referenziare da shell, dato
    che '[...]' e' un pattern di glob) -- l'etichetta leggibile nel titolo
    del grafico resta invece quella di _variance_label()."""
    m = _CUSTOM_RANGE_RE.match(str(variance))
    if m:
        return f"p{m.group(1)}-{m.group(2)}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(variance))


def load_solver_time_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Identica a load_solver_time_results di plot_solver_time_by_n.py:
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
            print(f"[plot_solver_time_by_n_low_p] rimosse {n_before - len(df)} righe "
                  f"duplicate (stesso instance_id in piu' file)")

    return df.sort_values(["m", "variance", "n"]).reset_index(drop=True)


def _available_methods(df: pd.DataFrame) -> List[tuple]:
    return [spec for spec in _METHOD_SPECS if spec[1] in df.columns]


def plot_solver_time_by_n_bars(df: pd.DataFrame, m_values: Optional[List[int]] = None,
                                variance_values: Optional[List[str]] = None,
                                agg: str = "median", log_scale: bool = True,
                                out_path: str = "results/solver_time_by_n_low_p.png") -> None:
    """Un file immagine per ciascuna combinazione (m, variance) presente nei
    dati (filtrata da m_values/variance_values se forniti). out_path e' usato
    come base, stesso principio di plot_solver_time_by_n.py.

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
    pairs["_variance_order"] = pairs["variance"].map(lambda v: _variance_sort_key(v)[0])
    pairs = pairs.sort_values(["m", "_variance_order", "variance"]).itertuples(index=False)
    for m, variance, _ in pairs:
        pair_sub = sub[(sub["m"] == m) & (sub["variance"] == variance)]
        pair_out = f"{root}_m{m}_{_variance_slug(variance)}{ext}"
        _plot_single_pair(pair_sub, m=m, variance=variance, methods=methods, agg=agg,
                           log_scale=log_scale, out_path=pair_out)


def _aggregate(rows: pd.DataFrame, runtime_col: str, agg: str) -> Optional[float]:
    values = rows[runtime_col].dropna()
    if values.empty:
        return None
    return float(values.median() if agg == "median" else values.mean())


def _limit_stats(rows: pd.DataFrame, runtime_col: str, status_col: Optional[str],
                  limit_status: Optional[str]) -> tuple:
    """Identica a _limit_stats di plot_solver_time_by_n.py: 'almeno una
    istanza al limite' fa scattare il tratteggio, l'etichetta 'k/n' mostra
    quante, tra le istanze con runtime_col non nullo, hanno effettivamente
    status_col == limit_status."""
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
        print(f"[plot_solver_time_by_n_low_p] nessun valore di n per m={m}, "
              f"variance={variance}, salto")
        return

    n_methods = len(methods)
    bar_width = (1.0 - _BAR_GAP) / n_methods
    group_positions = {n: i for i, n in enumerate(present_n)}
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
                continue
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
                y_text = height * 1.08 if log_scale else height
                ax.text(pos, y_text, frac_label, ha="center", va="bottom",
                        fontsize=7.5, color=_LIMIT_EDGE, clip_on=False)

    if log_scale:
        ax.set_yscale("log")

    if limit_hit_any:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 3 if log_scale else top + 0.12 * (top - bottom))

    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels([str(n) for n in present_n])
    ax.set_xlabel("numero di job")
    ax.set_ylabel(f"tempo di calcolo del solver (s{', scala log' if log_scale else ''})")
    ax.set_title(f"Tempo di calcolo dei solver al variare di n "
                 f"(m={m}, {_variance_label(variance)})")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = [Patch(facecolor=_METHOD_COLORS[name], label=name) for name, *_ in methods]
    if limit_hit_any:
        handles.append(Patch(facecolor="white", edgecolor=_LIMIT_EDGE, hatch=_LIMIT_HATCH,
                              label="time/memory limit raggiunto"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_solver_time_by_n_low_p] grafico salvato in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati (run_low_p_experiment.py, o "
                              "qualunque CSV compatibile di experiment_runner.py) da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=None,
                         help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--variance", nargs="+", type=str, default=None,
                         help="valori di variance da plottare (default: tutti quelli presenti "
                              "nei CSV; a differenza di plot_solver_time_by_n.py non e' "
                              "ristretto a low/high, vedi docstring del modulo)")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra piu' istanze della stessa classe "
                              "(m, n, variance, metodo) (default: median)")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false", default=True,
                         help="asse y in scala lineare invece che log (default: log)")
    parser.add_argument("--out", default="results/solver_time_by_n_low_p.png",
                         help="path base del file immagine di output: un file per ciascuna "
                              "combinazione (m, variance), vedi docstring del modulo")
    args = parser.parse_args()

    df = load_solver_time_results(args.csv)
    plot_solver_time_by_n_bars(df, m_values=args.m, variance_values=args.variance, agg=args.agg,
                               log_scale=args.log_scale, out_path=args.out)


if __name__ == "__main__":
    main()
