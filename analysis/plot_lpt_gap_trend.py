"""
Trend del gap % dell'euristica LPT rispetto al riferimento ottimo (MILP) al
variare di n -- pensato per i CSV prodotti da experiment_runner.py (servono
le colonne 'm', 'n', 'variance', 'instance_id' piu' la colonna di gap, per
default 'lpt_gap_rel_vs_ref'; vedi --gap-column piu' sotto). Il gap mediano
scende in modo pulito e monotono da qualche punto percentuale a n=10 fino a
ben sotto lo 0.1% a n=200, su tutti gli m e le varianze testate: una
convergenza che si legge a fatica dalle sole etichette percentuali sopra le
barre di un grafico a barre di valori assoluti (serve confrontare a memoria
6+ grafici diversi), ma diventa immediata con una linea per m su un unico
grafico.

Per ogni combinazione (m, n) vengono mostrate tutte le istanze come punti
(la dispersione tra seed e' spesso ampia, specie per n piccoli, quindi
mostrarla e' piu' onesto di una sola linea di mediana -- stesso principio
di plot_milp_scaling.py) piu' una linea di aggregato (mediana di default,
vedi --agg) per n che rende leggibile il trend.

Solo le istanze con un riferimento realmente ottimo vengono incluse: se la
colonna 'reference_type' e' presente, vengono tenute solo le righe con
reference_type == "MILP_OPTIMAL" (un gap calcolato contro un riferimento
subottimo, es. un MILP fermato al time limit, sottostimerebbe il vero gap
di LPT). Le righe scartate per questo motivo vengono conteggiate e
riportate nel riepilogo testuale, non silenziosamente ignorate.

L'asse y usa una scala 'symlog' (lineare vicino a zero, poi logaritmica
oltre --linthresh, default 0.01 punti percentuali) invece di una scala log
pura: il gap puo' essere ESATTAMENTE zero (LPT trova l'ottimo, capita non
di rado per n piccoli) e una scala log pura non puo' rappresentare lo zero.
La scala symlog mostra comunque la convergenza su piu' ordini di grandezza
per i valori diversi da zero, senza dover scartare o alterare i punti a
gap nullo.

Per default viene prodotto un grafico distinto per ciascun valore di
'variance' (stesso principio di plot_milp_scaling_by_variance): usare
--combined per un unico grafico con entrambe le varianze.

Uso da linea di comando (piu' CSV vengono concatenati, utile per unire
campagne con range di n complementari, come experiment1.csv + experiment2.csv):

    python analysis/plot_lpt_gap_trend.py \
        --csv results/experiment1.csv results/experiment2.csv \
        --out results/lpt_gap_trend.png

produce results/lpt_gap_trend_low.png e results/lpt_gap_trend_high.png.

Oppure programmaticamente:

    from analysis.plot_lpt_gap_trend import load_lpt_gap_results, plot_lpt_gap_trend_by_variance
    df = load_lpt_gap_results(["results/experiment1.csv", "results/experiment2.csv"])
    plot_lpt_gap_trend_by_variance(df, m_values=[2, 3, 5], out_path="results/lpt_gap_trend.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance"}

# stessa palette fissa per m usata in plot_milp_scaling.py / plot_dp_peak_memory.py
_COLORS = {2: "tab:green", 3: "tab:blue", 5: "tab:orange"}
_FALLBACK_COLORS = ["tab:purple", "tab:brown", "tab:red", "tab:cyan"]


def load_lpt_gap_results(csv_paths: Iterable[str], gap_column: str = "lpt_gap_rel_vs_ref") -> pd.DataFrame:
    """Concatena i CSV indicati e rimuove eventuali righe duplicate (stesso
    instance_id) nel caso gli stessi punti compaiano in piu' file, stesso
    principio di load_milp_results / load_solver_time_results."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = (REQUIRED_BASE_COLUMNS | {gap_column}) - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_lpt_gap_trend] rimosse {n_before - len(df)} righe duplicate "
                  f"(stesso instance_id in piu' file)")

    return df.sort_values(["m", "n", "variance"]).reset_index(drop=True)


def _filter_valid_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Tiene solo le righe con un riferimento realmente ottimo
    (reference_type == "MILP_OPTIMAL"), se la colonna e' presente -- vedi
    docstring del modulo. Se la colonna non e' presente, tutte le righe
    vengono tenute cosi' come sono (nessun modo di distinguere)."""
    if "reference_type" not in df.columns:
        return df
    n_before = len(df)
    out = df[df["reference_type"] == "MILP_OPTIMAL"]
    n_removed = n_before - len(out)
    if n_removed:
        print(f"[plot_lpt_gap_trend] escluse {n_removed} righe con reference_type diverso da "
              f"'MILP_OPTIMAL' (riferimento non ottimo, gap non affidabile)")
    return out


def plot_lpt_gap_trend(df: pd.DataFrame, m_values: List[int], gap_column: str = "lpt_gap_rel_vs_ref",
                        agg: str = "median", linthresh: float = 0.01,
                        out_path: str = "results/lpt_gap_trend.png",
                        title_suffix: str = "") -> None:
    """Grafico (x log, y symlog) del gap % di LPT vs n, una serie per
    ciascun m in m_values: tutte le istanze come punti piu' una linea di
    aggregato (agg: "median" default, o "mean") per n -- stesso schema di
    plot_milp_scaling.py.

    linthresh: soglia (in punti percentuali) sotto la quale l'asse y passa
    da logaritmico a lineare (per rappresentare correttamente i gap
    esattamente nulli, vedi docstring del modulo)."""
    if agg not in ("median", "mean"):
        raise ValueError(f"agg deve essere 'median' o 'mean' (ricevuto: {agg!r})")

    df = df[df["m"].isin(m_values)]
    if df.empty:
        raise ValueError(f"nessuna riga per m in {m_values} nei dati forniti")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, m in enumerate(sorted(m_values)):
        sub = df[df["m"] == m]
        if sub.empty:
            continue
        color = _COLORS.get(m, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        gap_pct = sub[gap_column] * 100.0

        ax.scatter(sub["n"], gap_pct, color=color, alpha=0.4, s=26, edgecolors="none",
                   label=f"m={m} (istanze)")

        agg_by_n = (sub.assign(_gap_pct=gap_pct).groupby("n")["_gap_pct"]
                    .agg(agg).sort_index())
        ax.plot(agg_by_n.index, agg_by_n.values, color=color, linewidth=2,
                marker="o", markersize=5, label=f"m={m} ({agg})")

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("n (numero di job)")
    ax.set_ylabel(f"gap LPT vs ottimo (%, scala symlog, soglia lineare {linthresh:g}%)")
    ax.set_title(f"Trend del gap % di LPT al variare di n{title_suffix}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_lpt_gap_trend] grafico salvato in {out_path}")

    _print_gap_summary(df, m_values, gap_column, agg)


def plot_lpt_gap_trend_by_variance(df: pd.DataFrame, m_values: List[int],
                                    gap_column: str = "lpt_gap_rel_vs_ref", agg: str = "median",
                                    linthresh: float = 0.01,
                                    out_path: str = "results/lpt_gap_trend.png") -> None:
    """Come plot_lpt_gap_trend, ma produce un file immagine distinto per
    ciascun valore di 'variance', stesso principio di
    plot_milp_scaling_by_variance. out_path e' usato come base:
    'results/lpt_gap_trend.png' diventa 'results/lpt_gap_trend_low.png' e
    'results/lpt_gap_trend_high.png'."""
    if "variance" not in df.columns:
        raise ValueError("colonna 'variance' assente: impossibile separare i grafici per "
                          "varianza (usare plot_lpt_gap_trend per un grafico unico)")

    root, ext = os.path.splitext(out_path)
    for variance in sorted(df["variance"].dropna().unique()):
        sub = df[df["variance"] == variance]
        variance_out = f"{root}_{variance}{ext}"
        plot_lpt_gap_trend(sub, m_values=m_values, gap_column=gap_column, agg=agg,
                           linthresh=linthresh, out_path=variance_out,
                           title_suffix=f" (varianza {variance})")


def _print_gap_summary(df: pd.DataFrame, m_values: List[int], gap_column: str, agg: str) -> None:
    g = (df[df["m"].isin(m_values)].assign(_gap_pct=df[gap_column] * 100.0)
         .groupby(["m", "n"])["_gap_pct"].agg(agg))
    print(f"[plot_lpt_gap_trend] gap % di LPT ({agg}) per (m, n):")
    print(g.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati (experiment_runner.py) da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=[2, 3, 5],
                         help="valori di m da mettere a confronto (default: 2 3 5)")
    parser.add_argument("--gap-column", default="lpt_gap_rel_vs_ref",
                         help="colonna del gap relativo da plottare, come frazione (non "
                              "percentuale: viene moltiplicata per 100 internamente). Default: "
                              "'lpt_gap_rel_vs_ref'; usare 'dp_gap_rel_vs_ref' per il gap della DP "
                              "invece che di LPT, se presente nel CSV")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra le istanze della stessa classe (m, n) per la linea "
                              "di trend (default: median)")
    parser.add_argument("--linthresh", type=float, default=0.01,
                         help="soglia (in punti percentuali) sotto la quale l'asse y passa da "
                              "logaritmico a lineare, per rappresentare i gap esattamente nulli "
                              "(default: 0.01)")
    parser.add_argument("--out", default="results/lpt_gap_trend.png",
                         help="path base del file immagine di output (con --by-variance, usato "
                              "come base per un file per varianza: vedi docstring del modulo)")
    parser.add_argument("--by-variance", dest="by_variance", action="store_true", default=True,
                         help="un grafico distinto per ciascun valore di 'variance' (default)")
    parser.add_argument("--combined", dest="by_variance", action="store_false",
                         help="un unico grafico con tutte le varianze insieme")
    args = parser.parse_args()

    df = load_lpt_gap_results(args.csv, gap_column=args.gap_column)
    df = _filter_valid_reference(df)

    if args.by_variance:
        plot_lpt_gap_trend_by_variance(df, m_values=args.m, gap_column=args.gap_column,
                                       agg=args.agg, linthresh=args.linthresh, out_path=args.out)
    else:
        plot_lpt_gap_trend(df, m_values=args.m, gap_column=args.gap_column, agg=args.agg,
                           linthresh=args.linthresh, out_path=args.out)


if __name__ == "__main__":
    main()
