"""
Andamento del tempo di calcolo del MILP (Gurobi) al variare di n, per uno o
piu' valori di m fissati -- pensato per i CSV prodotti da experiment_runner.py
(o da campagne di calibrazione ad-hoc con le stesse colonne: servono solo
'n', 'm', 'milp_status', 'milp_runtime').

Evidenzia le istanze in cui il MILP non ha chiuso entro il time limit
(milp_status == "TIME_LIMIT"): sono marcate sul grafico e, per queste,
milp_runtime e' il tempo speso fino al time limit (non un tempo di
risoluzione all'ottimo), quindi vanno lette come "runtime troncato al
limite", non come punti comparabili agli altri sulla stessa curva.

Per default viene prodotto un grafico distinto per ciascun valore di
'variance' (basso/alto), non un unico grafico con tutto insieme: la
dispersione del runtime dipende fortemente dalla varianza dei tempi di
lavorazione (vedi calibrate_dp_lpt.py per lo stesso principio applicato
alla DP), quindi mescolarle nello stesso grafico confonderebbe due regimi
diversi. Usare --combined per un unico grafico con tutte le varianze.

Uso da linea di comando (piu' CSV vengono concatenati, utile quando la
campagna e' stata spezzata su piu' run/n-range, come qui):

    python analysis/plot_milp_scaling.py \\
        --csv results/calib_bump.csv results/calib_sweep.csv results/crash_milp.csv \\
        --m 3 5 --time-limit 120 --out results/milp_runtime_scaling.png

produce results/milp_runtime_scaling_low.png e
results/milp_runtime_scaling_high.png.

Oppure programmaticamente:

    from analysis.plot_milp_scaling import load_milp_results, plot_milp_scaling_by_variance
    df = load_milp_results(["results/calib_bump.csv", "results/calib_sweep.csv"])
    plot_milp_scaling_by_variance(df, m_values=[3, 5], time_limit=120,
                                   out_path="results/milp_runtime_scaling.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {"n", "m", "milp_status", "milp_runtime"}

# palette fissa per m: cosi' lo stesso m ha sempre lo stesso colore anche
# confrontando grafici prodotti in run diverse
_COLORS = {2: "tab:green", 3: "tab:blue", 5: "tab:orange"}
_FALLBACK_COLORS = ["tab:purple", "tab:brown", "tab:red", "tab:cyan"]


def load_milp_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Concatena i CSV indicati (es. calib_bump.csv + calib_sweep.csv +
    crash_milp.csv, che coprono range di n complementari) e rimuove eventuali
    istanze duplicate (stesso instance_id) nel caso gli stessi punti compaiano
    in piu' di un file."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_milp_scaling] rimosse {n_before - len(df)} righe duplicate "
                  f"(stesso instance_id in piu' file)")

    return df.sort_values(["m", "n"]).reset_index(drop=True)


def plot_milp_scaling(df: pd.DataFrame, m_values: List[int], time_limit: Optional[float] = 120,
                       out_path: str = "results/milp_runtime_scaling.png",
                       title_suffix: str = "") -> None:
    """Grafico log-log di milp_runtime vs n, una serie per ciascun m in
    m_values. Per ogni m: tutte le istanze come punti (la varianza tra seed/
    variance a parita' di n e' spesso ampia, specie per m piu' grandi -- vedi
    modulo docstring -- quindi mostrarla e' piu' onesto di una sola curva di
    mediana) piu' una linea di mediana per n che rende leggibile il trend.
    La mediana e' calcolata su tutte le istanze, comprese quelle con
    milp_status == "TIME_LIMIT": escluderle lascerebbe un buco nella linea
    ogni volta che TUTTE le istanze di un dato n raggiungono il time limit
    (es. m=5, n=2e6), quindi il loro runtime (troncato al limite) entra nella
    mediana come stima per difetto del tempo di risoluzione reale. Il
    riepilogo testuale stampato a fine funzione elenca comunque quali
    istanze hanno raggiunto il time limit, per non perdere l'informazione."""
    df = df[df["m"].isin(m_values)]
    if df.empty:
        raise ValueError(f"nessuna riga per m in {m_values} nei dati forniti")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, m in enumerate(sorted(m_values)):
        sub = df[df["m"] == m]
        color = _COLORS.get(m, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])

        ax.scatter(sub["n"], sub["milp_runtime"], color=color, alpha=0.45,
                   s=28, edgecolors="none", label=f"m={m} (istanze)")

        median_by_n = sub.groupby("n")["milp_runtime"].median().sort_index()
        ax.plot(median_by_n.index, median_by_n.values, color=color, linewidth=2,
                marker="o", markersize=5, label=f"m={m} (mediana)")

    if time_limit is not None:
        ax.axhline(time_limit, color="gray", linestyle="--", linewidth=1.2,
                   label=f"time limit ({time_limit:.0f}s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("numero di job")
    ax.set_ylabel("tempo di calcolo MIP (s, scala log)")
    ax.set_title(f"Tempo di calcolo del MIP al variare di n{title_suffix}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_milp_scaling] grafico salvato in {out_path}")

    _print_time_limit_summary(df, m_values)


def plot_milp_scaling_by_variance(df: pd.DataFrame, m_values: List[int],
                                   time_limit: Optional[float] = 120,
                                   out_path: str = "results/milp_runtime_scaling.png") -> None:
    """Come plot_milp_scaling, ma produce un file immagine distinto per
    ciascun valore della colonna 'variance' (tipicamente 'low' e 'high'),
    invece di un unico grafico con entrambe mescolate. out_path e' usato come
    base: 'results/milp_runtime_scaling.png' diventa
    'results/milp_runtime_scaling_low.png' e
    'results/milp_runtime_scaling_high.png'."""
    if "variance" not in df.columns:
        raise ValueError("colonna 'variance' assente: impossibile separare i grafici per "
                          "varianza (usare plot_milp_scaling per un grafico unico)")

    root, ext = os.path.splitext(out_path)
    for variance in sorted(df["variance"].dropna().unique()):
        sub = df[df["variance"] == variance]
        variance_out = f"{root}_{variance}{ext}"
        plot_milp_scaling(sub, m_values=m_values, time_limit=time_limit, out_path=variance_out,
                           title_suffix=f" (varianza {variance})")


def _print_time_limit_summary(df: pd.DataFrame, m_values: List[int]) -> None:
    hit = df[(df["m"].isin(m_values)) & (df["milp_status"] == "TIME_LIMIT")]
    if hit.empty:
        print("[plot_milp_scaling] nessuna istanza ha raggiunto il time limit "
              f"per m in {m_values}")
        return
    print(f"[plot_milp_scaling] istanze che hanno raggiunto il time limit (m in {m_values}):")
    cols = [c for c in ["instance_id", "m", "n", "variance", "milp_runtime", "milp_gap_rel"]
            if c in hit.columns]
    print(hit[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati MILP da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=[3, 5],
                         help="valori di m da mettere a confronto (default: 3 5)")
    parser.add_argument("--time-limit", type=float, default=120.0,
                         help="time limit del MILP in secondi, per la linea di riferimento "
                              "(default: 120; usare --time-limit -1 per non disegnarla)")
    parser.add_argument("--out", default="results/milp_runtime_scaling.png",
                         help="path del file immagine di output (con --by-variance, usato come "
                              "base per un file per varianza: vedi docstring del modulo)")
    parser.add_argument("--by-variance", dest="by_variance", action="store_true", default=True,
                         help="un grafico distinto per ciascun valore di 'variance' (default)")
    parser.add_argument("--combined", dest="by_variance", action="store_false",
                         help="un unico grafico con tutte le varianze insieme")
    args = parser.parse_args()

    df = load_milp_results(args.csv)
    time_limit = None if args.time_limit < 0 else args.time_limit
    if args.by_variance:
        plot_milp_scaling_by_variance(df, m_values=args.m, time_limit=time_limit, out_path=args.out)
    else:
        plot_milp_scaling(df, m_values=args.m, time_limit=time_limit, out_path=args.out)


if __name__ == "__main__":
    main()
