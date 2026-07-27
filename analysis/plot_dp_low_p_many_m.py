"""
Grafici per il CSV di dettaglio prodotto da calibrate_dp_low_p_many_m.py
(colonne attese: m, dp_status, dp_runtime, dp_peak_vsz_mb, dp_states,
reference_cmax/lpt_cmax). Serve solo il CSV di dettaglio: i tre grafici
qui sotto ricalcolano da soli le stesse aggregazioni del CSV di riepilogo
(_summary.csv), quindi quest'ultimo non e' richiesto come input.

Nota sulla metrica di memoria: il grafico della memoria di picco usa
dp_peak_vsz_mb (VmPeak, memoria VIRTUALE), NON dp_peak_memory_mb (RSS,
memoria residente/fisica), anche se quest'ultima e' comunque presente nel
CSV. Il motivo e' che il budget --memory-limit-gb passato alla DP e'
imposto via RLIMIT_AS, che limita lo spazio di indirizzi VIRTUALE, non la
RSS (vedi dp_solver.py, _set_memory_limit / _peak_vsz_mb): un grafico in
RSS mostra le istanze MEMORY_LIMIT_EXCEEDED fermarsi ben al di sotto del
budget (l'allocatore riserva/frammenta piu' spazio di quanto ne scriva
davvero, specie durante il rehashing dei `set` Python nella rappresenta-
zione sparsa), il che sembra un'incongruenza se non si sa che il limite
non e' sulla RSS. In VmPeak, invece, le istanze censurate atterrano vicine
alla linea del budget, perche' e' esattamente la metrica che ha fatto
scattare il limite.

Punto delicato, comune a tutti e tre i grafici: quando dp_status ==
"MEMORY_LIMIT_EXCEEDED" i valori misurati sono CENSURATI, non falliti a
zero:
  - dp_runtime e' il tempo trascorso PRIMA di esaurire la memoria, non un
    tempo-a-soluzione (la DP non ha chiuso). Non e' quindi direttamente
    comparabile al dp_runtime delle righe OPTIMAL (quello si', e' tempo
    all'ottimo).
  - dp_peak_vsz_mb e' quanto spazio virtuale e' stato effettivamente
    riservato prima del crash, MA il vero fabbisogno per arrivare
    all'ottimo e' ignoto e >= questo valore (limitato dall'alto solo dal
    budget passato a --memory-limit-gb) -- e' un "almeno cosi' tanto", non
    "esattamente cosi' tanto".
  - dp_states e' -1 quando la DP esaurisce la memoria PRIMA di riuscire a
    contare gli stati dell'ultimo livello: va trattato come mancante, non
    come "zero stati".
  - dp_cmax nelle righe MEMORY_LIMIT_EXCEEDED (se non-null) e' l'incumbent
    parziale sui soli job elaborati prima del crash (vedi dp_solver.py):
    NON e' un upper bound ammissibile per l'istanza intera, quindi qui
    non viene mai plottato come "soluzione trovata" -- il valore di
    riferimento usato ovunque per il Cmax e' invece reference_cmax/
    lpt_cmax, sempre disponibile e sempre ammissibile.
Per questo tutti e tre i grafici distinguono esplicitamente le righe
OPTIMAL (misura piena) dalle MEMORY_LIMIT_EXCEEDED (misura censurata),
con marker/colore diversi e una nota a legenda, invece di trattarle come
punti equivalenti sulla stessa curva.

Uso da linea di comando:

    python analysis/plot_dp_low_p_many_m.py \
        --csv results/calibration_dp_low_p_many_m.csv \
        --memory-limit-gb 16 \
        --out-dir results/grafici_low_p_many_m

produce tre file PNG in --out-dir:
  1_value_vs_cost.png        Cmax (piatto) vs dp_runtime (log) per m
  2_memory_and_states.png    memoria di picco (virtuale, VmPeak) e stati raggiunti per m
  3_success_rate.png         frazione di istanze chiuse OPTIMAL per m
"""

from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"m", "dp_status", "dp_runtime", "dp_peak_vsz_mb", "dp_states"}

_OPTIMAL_COLOR = "tab:blue"
_CENSORED_COLOR = "tab:red"
_CMAX_COLOR = "tab:green"


def load_detail(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti in {csv_path}: {sorted(missing)}")
    if "reference_cmax" not in df.columns and "lpt_cmax" not in df.columns:
        raise ValueError(f"serve almeno una tra 'reference_cmax' e 'lpt_cmax' in {csv_path}")
    df["cmax_ref"] = df["reference_cmax"] if "reference_cmax" in df.columns else df["lpt_cmax"]
    df["is_optimal"] = df["dp_status"] == "OPTIMAL"
    return df.sort_values("m").reset_index(drop=True)


def plot_value_vs_cost(df: pd.DataFrame, out_path: str) -> None:
    """Grafico principale: sullo stesso asse x (m) mostra che il VALORE
    della soluzione (Cmax di riferimento, asse sx, lineare) resta piatto
    mentre il COSTO per ottenerlo con la DP (dp_runtime, asse dx, log)
    esplode. Le istanze censurate (OOM) sono marcate a parte sull'asse dx,
    col tempo-a-fallimento invece del tempo-a-soluzione."""
    fig, ax_cmax = plt.subplots(figsize=(8, 5.5))
    ax_time = ax_cmax.twinx()

    m_values = sorted(df["m"].unique())
    jitter = {m: i * 0.06 - 0.06 for i, m in enumerate(range(3))}  # piccolo offset orizzontale

    # Cmax di riferimento: sempre disponibile e ammissibile (LPT), un punto
    # per istanza + media per m, cosi' si vede sia il livello sia la variabilita'
    for m in m_values:
        sub = df[df["m"] == m]
        xs = [m + (k % 3) * 0.08 - 0.08 for k in range(len(sub))]
        ax_cmax.scatter(xs, sub["cmax_ref"], color=_CMAX_COLOR, alpha=0.5, zorder=2, s=28)
    means_cmax = df.groupby("m")["cmax_ref"].mean()
    ax_cmax.plot(means_cmax.index, means_cmax.values, color=_CMAX_COLOR, marker="o",
                 linewidth=2, zorder=3, label="Cmax di riferimento")
    if "target_cmax" in df.columns:
        t = df["target_cmax"].iloc[0]
        ax_cmax.axhspan(t * 0.8, t * 1.2, color=_CMAX_COLOR, alpha=0.08, zorder=1)

    ax_cmax.set_xlabel("numero di macchine")
    ax_cmax.set_ylabel("Cmax di riferimento", color=_CMAX_COLOR)
    ax_cmax.tick_params(axis="y", labelcolor=_CMAX_COLOR)
    ax_cmax.set_xticks(m_values)
    ax_cmax.set_ylim(0, max(df["cmax_ref"].max() * 1.3, 10))

    # dp_runtime: OPTIMAL = tempo a soluzione (misura piena); MEMORY_LIMIT_EXCEEDED
    # = tempo al crash (misura censurata) -- marker distinto, mai la stessa curva
    opt = df[df["is_optimal"]]
    cens = df[~df["is_optimal"]]
    if not opt.empty:
        ax_time.scatter(opt["m"], opt["dp_runtime"], color=_OPTIMAL_COLOR, marker="o",
                         s=50, zorder=4, label="dp_runtime (OPTIMAL)")
        means_t = opt.groupby("m")["dp_runtime"].mean()
        ax_time.plot(means_t.index, means_t.values, color=_OPTIMAL_COLOR, linewidth=2, zorder=3)
    if not cens.empty:
        ax_time.scatter(cens["m"], cens["dp_runtime"], color=_CENSORED_COLOR, marker="x",
                         s=70, linewidth=2.2, zorder=4,
                         label="dp_runtime (MEMORY_LIMIT_EXCEEDED)")

    ax_time.set_yscale("log")
    ax_time.set_ylabel("dp_runtime in secondi, scala log", color=_OPTIMAL_COLOR)
    ax_time.tick_params(axis="y", labelcolor=_OPTIMAL_COLOR)

    ax_cmax.set_title("Valore della soluzione costante vs costo della DP che esplode")
    h1, l1 = ax_cmax.get_legend_handles_labels()
    h2, l2 = ax_time.get_legend_handles_labels()
    ax_cmax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    ax_cmax.grid(True, axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_low_p_many_m] salvato {out_path}")


def plot_memory_and_states(df: pd.DataFrame, out_path: str,
                            memory_limit_gb: float | None = None) -> None:
    """Due pannelli affiancati: memoria di picco e numero di stati, entrambi
    in scala log rispetto a m. I punti censurati (OOM) sono marcati a parte;
    per gli stati, le righe con dp_states == -1 (non contati prima del
    crash) sono escluse dalla curva e segnalate a parte con una freccia,
    non disegnate come se fossero un valore noto.

    La memoria di picco e' dp_peak_vsz_mb (VmPeak, virtuale), non la RSS:
    e' la metrica su cui agisce RLIMIT_AS, quindi e' quella che si confronta
    in modo sensato con la linea del budget --memory-limit-gb (vedi nota nel
    docstring del modulo)."""
    fig, (ax_mem, ax_states) = plt.subplots(1, 2, figsize=(12, 5.5))

    opt = df[df["is_optimal"]]
    cens = df[~df["is_optimal"]]

    # --- memoria di picco (virtuale, VmPeak) ---
    if not opt.empty:
        ax_mem.scatter(opt["m"], opt["dp_peak_vsz_mb"], color=_OPTIMAL_COLOR, marker="o",
                        s=50, zorder=3, label="OPTIMAL")
        means = opt.groupby("m")["dp_peak_vsz_mb"].mean()
        ax_mem.plot(means.index, means.values, color=_OPTIMAL_COLOR, linewidth=2, zorder=2)
    if not cens.empty:
        ax_mem.scatter(cens["m"], cens["dp_peak_vsz_mb"], color=_CENSORED_COLOR, marker="^",
                        s=70, zorder=3,
                        label="MEMORY_LIMIT_EXCEEDED")
    if memory_limit_gb is not None:
        ax_mem.axhline(memory_limit_gb * 1024, color="gray", linestyle="--", linewidth=1.2,
                        label=f"budget ({memory_limit_gb:.0f}GB)")
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("numero di macchine")
    ax_mem.set_ylabel("memoria virtuale, scala log")
    ax_mem.set_title("Memoria di picco")
    ax_mem.set_xticks(sorted(df["m"].unique()))
    ax_mem.legend(fontsize=8)
    ax_mem.grid(True, which="both", linestyle=":", alpha=0.4)

    # --- numero di stati ---
    states_known = df[df["dp_states"] >= 0]
    states_unknown_ms = sorted(df.loc[df["dp_states"] < 0, "m"].unique())
    opt_s = states_known[states_known["is_optimal"]]
    if not opt_s.empty:
        ax_states.scatter(opt_s["m"], opt_s["dp_states"], color=_OPTIMAL_COLOR, marker="o",
                           s=50, zorder=3, label="OPTIMAL")
        means = opt_s.groupby("m")["dp_states"].mean()
        ax_states.plot(means.index, means.values, color=_OPTIMAL_COLOR, linewidth=2, zorder=2)
    ax_states.set_yscale("log")
    ax_states.set_xlabel("numero di macchine")
    ax_states.set_ylabel("stati raggiunti, scala log")
    ax_states.set_title("Stati della DP")
    ax_states.set_xticks(sorted(df["m"].unique()))
    if states_unknown_ms and not opt_s.empty:
        y_top = states_known["dp_states"].max() * 3
        for m in states_unknown_ms:
            ax_states.annotate("non contati\n(OOM prima\ndel conteggio)",
                                xy=(m, y_top), xytext=(m, y_top), ha="center", va="bottom",
                                fontsize=8, color=_CENSORED_COLOR)
            ax_states.scatter([m], [y_top], color=_CENSORED_COLOR, marker="^", s=70, zorder=3)
    ax_states.legend(fontsize=8)
    ax_states.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_low_p_many_m] salvato {out_path}")


def plot_success_rate(df: pd.DataFrame, out_path: str) -> None:
    """Frazione di istanze chiuse OPTIMAL per ciascun m: la 'scogliera'
    (cliff) tra il regime in cui la DP chiude sempre e quello in cui non
    chiude mai piu', a parita' di Cmax atteso."""
    frac = df.groupby("m")["is_optimal"].mean()
    n_per_m = df.groupby("m").size()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [_OPTIMAL_COLOR if f > 0 else _CENSORED_COLOR for f in frac.values]
    ax.bar(frac.index.astype(str), frac.values, color=colors)
    for m, f, n in zip(frac.index, frac.values, n_per_m.values):
        ax.text(str(m), f + 0.03, f"{int(round(f * n))}/{n}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("numero di macchine")
    ax.set_ylabel("frazione istanze OPTIMAL")
    ax.set_title("Quota di istanze risolte all'ottimo entro il budget, per m")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_low_p_many_m] salvato {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="CSV di dettaglio da calibrate_dp_low_p_many_m.py")
    ap.add_argument("--memory-limit-gb", type=float, default=16.0,
                     help="budget usato nella run (per la linea di riferimento nel grafico memoria)")
    ap.add_argument("--out-dir", default="results/grafici_low_p_many_m")
    args = ap.parse_args()

    df = load_detail(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    plot_value_vs_cost(df, out_path=os.path.join(args.out_dir, "1_value_vs_cost.png"))
    plot_memory_and_states(df, out_path=os.path.join(args.out_dir, "2_memory_and_states.png"),
                            memory_limit_gb=args.memory_limit_gb)
    plot_success_rate(df, out_path=os.path.join(args.out_dir, "3_success_rate.png"))


if __name__ == "__main__":
    main()
