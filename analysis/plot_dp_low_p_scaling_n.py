"""
Grafici per i CSV di dettaglio prodotti da calibrate_dp_low_p_scaling_n.py
(crescita a raddoppio di n, a m fisso) e calibrate_dp_low_p_sweep_n.py
(rifinitura tra l'ultimo n ok e il primo n fallito) -- pensato per essere
alimentato con ENTRAMBI insieme: lo scaling da' la vista larga a grana
grossa, lo sweep infittisce solo la zona di transizione. Stesso identico
schema di colonne per i due script (vedi il docstring di
calibrate_dp_low_p_sweep_n.py), quindi si concatenano senza bisogno di
adattamenti -- --sweep-csv e' comunque opzionale, il grafico funziona
anche solo con lo scaling.

Un file per ciascun m (out_path e' usato come base, stesso principio di
plot_dp_low_p_many_m.py e plot_dp_calibration_bars.py). Asse x = n
(scala log: la crescita a raddoppio copre facilmente 2-3 ordini di
grandezza), asse y = frazione delle istanze presenti a quell'n (su
instances-per-class, che puo' variare da un n all'altro se lo sweep ne
ha usate meno) che cadono in ciascuno dei tre esiti possibili:

  - OPTIMAL "pulito" (dp_status == OPTIMAL e stop_reason is None): la DP
    ha chiuso entro budget di tempo E memoria;
  - MEMORY_LIMIT_EXCEEDED: vero fallimento di memoria;
  - TIME_SOFT_CAP_STOPPED (dp_status resta OPTIMAL, il segnale e' in
    stop_reason): NON e' un fallimento della DP, e' solo il soft cap di
    tempo scelto in calibrazione. Va tenuto visivamente separato da
    MEMORY_LIMIT_EXCEEDED per lo stesso motivo per cui
    calibrate_dp_low_p_sweep_n.py lo esclude di default dai bracket da
    affinare (vedi quel modulo): confonderli farebbe sembrare "limite di
    memoria" qualcosa che e' solo una soglia di pazienza della
    calibrazione, non una proprieta' della DP.

Le tre frazioni sommano a 1 per costruzione SOLO se ogni istanza cade in
uno dei tre esiti attesi. Se dp_status == "ERROR" compare (fallimento
diverso da OOM, vedi dp_solver.py), quelle righe vengono escluse dalle
frazioni disegnate e segnalate a parte con un avviso invece di essere
fatte sparire silenziosamente nell'arrotondamento.

I punti provenienti dal CSV di scaling e quelli dal CSV di sweep sono
disegnati con lo stesso colore/serie (stesso esito = stesso colore) ma
marker PIENO per lo scaling e marker VUOTO per lo sweep, cosi' si vede a
colpo d'occhio dove la risoluzione e' stata infittita.

Se calcolabile per quell'm (stessa identica logica di _build_summary in
calibrate_dp_low_p_scaling_n.py, riusata qui via import diretto invece di
duplicarla), il bracket [ultimo n con >=1 OPTIMAL pulito, primo n con 0]
viene evidenziato con una fascia verticale grigia tratteggiata -- e' lo
stesso bracket che calibrate_dp_low_p_sweep_n.py userebbe per decidere
cosa affinare, quindi il grafico e la scelta di quali n sweepare restano
sempre coerenti tra loro. Con --representation both il bracket non viene
disegnato (sparse e dense avrebbero bracket distinti sullo stesso m, la
fascia unica sarebbe fuorviante): il resto del grafico funziona comunque.

Uso:
    python analysis/plot_dp_low_p_scaling_n.py \\
        --scaling-csv results/calibration_dp_low_p_scaling_n.csv \\
        --sweep-csv results/calibration_dp_low_p_sweep_n.csv \\
        --out results/dp_low_p_scaling_n.png

--sweep-csv accetta piu' file (per concatenare piu' run di sweep, es. uno
per modalita' o uno per m diversi lanciati separatamente).
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from calibrate_dp_low_p_scaling_n import _build_summary  # riuso diretto, vedi docstring

REQUIRED_COLUMNS = {"m", "n", "seed", "dp_status", "stop_reason"}

_COLORS = {"optimal": "tab:green", "oom": "firebrick", "time_cap": "tab:orange"}
_LABELS = {"optimal": "OPTIMAL (pulito)", "oom": "MEMORY_LIMIT_EXCEEDED",
           "time_cap": "TIME_SOFT_CAP_STOPPED"}
_MARKERS = {"optimal": "o", "oom": "^", "time_cap": "s"}


def _nan_to_none(records: List[dict]) -> List[dict]:
    # pd.read_csv riporta le celle vuote (es. stop_reason quando assente) come
    # NaN float; _build_summary si aspetta "is None", non isna(), quindi si
    # sostituisce esplicitamente invece di affidarsi al cast implicito di
    # df.where() (che su colonne miste spesso NON converte, vedi
    # calibrate_dp_low_p_sweep_n.py dove lo stesso bug e' stato trovato e
    # corretto).
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


def _outcome(row: dict) -> str:
    if row["dp_status"] == "MEMORY_LIMIT_EXCEEDED":
        return "oom"
    if row["dp_status"] == "OPTIMAL":
        return "time_cap" if row["stop_reason"] == "TIME_SOFT_CAP_STOPPED" else "optimal"
    return "error"  # dp_status == "ERROR" o qualunque valore inatteso


def load_scaling_sweep_results(scaling_csv: str, sweep_csvs: Iterable[str]) -> pd.DataFrame:
    """Carica il CSV di scaling (obbligatorio) e uno o piu' CSV di sweep
    (opzionali), aggiunge una colonna 'source' ('scaling'/'sweep') per poter
    distinguere i marker nel grafico, concatena e rimuove eventuali righe
    duplicate (stessa (m, n, seed) in piu' file)."""
    df_scaling = pd.read_csv(scaling_csv)
    df_scaling["source"] = "scaling"
    frames = [df_scaling]
    for p in sweep_csvs or []:
        df_s = pd.read_csv(p)
        df_s["source"] = "sweep"
        frames.append(df_s)
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    n_before = len(df)
    df = df.drop_duplicates(subset=["m", "n", "seed"], keep="first")
    if len(df) < n_before:
        print(f"[plot_dp_low_p_scaling_n] rimosse {n_before - len(df)} righe duplicate "
              f"(stessa (m, n, seed) in piu' file)")

    return df.sort_values(["m", "n"]).reset_index(drop=True)


def plot_scaling_sweep(df: pd.DataFrame, m_values: List[int],
                        representation_mode: str = "auto",
                        out_path: str = "results/dp_low_p_scaling_n.png") -> None:
    bracket_by_m = {}
    if representation_mode != "both":
        records = _nan_to_none(df.to_dict("records"))
        summary = _build_summary(records, representation_mode)
        bracket_by_m = {s["m"]: (s["last_n_some_optimal"], s["first_n_all_failed"]) for s in summary}
    else:
        print("[plot_dp_low_p_scaling_n] --representation both: bracket non disegnato "
              "(sparse/dense avrebbero bracket distinti sullo stesso m), vedi docstring")

    for m in sorted(m_values):
        sub = df[df["m"] == m]
        if sub.empty:
            print(f"[plot_dp_low_p_scaling_n] nessuna riga per m={m}, salto")
            continue
        root, ext = os.path.splitext(out_path)
        _plot_single_m(sub, m=m, bracket=bracket_by_m.get(m), out_path=f"{root}_m{m}{ext}")


def _plot_single_m(sub: pd.DataFrame, m: int, bracket: Optional[tuple], out_path: str) -> None:
    rows = _nan_to_none(sub.to_dict("records"))
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    ns = sorted(by_n.keys())

    frac = {"optimal": [], "oom": [], "time_cap": []}
    is_sweep_point = []
    n_errors_total = 0
    for n in ns:
        rs = by_n[n]
        total = len(rs)
        counts = {"optimal": 0, "oom": 0, "time_cap": 0, "error": 0}
        for r in rs:
            counts[_outcome(r)] += 1
        n_errors_total += counts["error"]
        for key in frac:
            frac[key].append(counts[key] / total)
        is_sweep_point.append(all(r.get("source") == "sweep" for r in rs))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for key in ("optimal", "oom", "time_cap"):
        ax.plot(ns, frac[key], color=_COLORS[key], linewidth=1.6, zorder=2,
                label=_LABELS[key])
        filled = [(n, y) for n, y, sw in zip(ns, frac[key], is_sweep_point) if not sw]
        hollow = [(n, y) for n, y, sw in zip(ns, frac[key], is_sweep_point) if sw]
        if filled:
            fx, fy = zip(*filled)
            ax.scatter(fx, fy, color=_COLORS[key], marker=_MARKERS[key], s=55, zorder=3)
        if hollow:
            hx, hy = zip(*hollow)
            ax.scatter(hx, hy, facecolors="none", edgecolors=_COLORS[key], marker=_MARKERS[key],
                       s=55, zorder=3, linewidths=1.4)

    if bracket and bracket[0] is not None and bracket[1] is not None:
        low, high = bracket
        ax.axvspan(low, high, color="gray", alpha=0.12, zorder=1)
        ax.axvline(low, color="gray", linestyle=":", linewidth=1, zorder=1)
        ax.axvline(high, color="gray", linestyle=":", linewidth=1, zorder=1)
        ax.text(math.sqrt(low * high), 1.045, "bracket", ha="center", va="bottom",
                fontsize=8, color="dimgray")

    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlabel("n (numero di job, scala log)")
    ax.set_ylabel("frazione di istanze")
    ax.set_title(f"DP: esito al variare di n (m={m})")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.4)

    handles, labels_ = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="o", color="gray", linestyle="", markerfacecolor="gray",
               markersize=7, label="da scaling (grana grossa)"),
        Line2D([0], [0], marker="o", color="gray", linestyle="", markerfacecolor="none",
               markersize=7, label="da sweep (rifinito)"),
    ]
    ax.legend(handles=handles, labels=labels_ + ["da scaling (grana grossa)", "da sweep (rifinito)"],
              fontsize=8, loc="center left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_low_p_scaling_n] salvato {out_path}")
    if n_errors_total:
        print(f"[plot_dp_low_p_scaling_n] attenzione: {n_errors_total} righe con dp_status=='ERROR' "
              f"per m={m}, escluse dalle frazioni disegnate (non rientrano nei tre esiti attesi)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling-csv", required=True,
                     help="CSV di dettaglio di calibrate_dp_low_p_scaling_n.py")
    ap.add_argument("--sweep-csv", nargs="*", default=[],
                     help="uno o piu' CSV di dettaglio di calibrate_dp_low_p_sweep_n.py (opzionale)")
    ap.add_argument("--m", nargs="+", type=int, default=None,
                     help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    ap.add_argument("--representation", choices=["auto", "sparse", "dense", "both"], default="auto",
                     help="deve corrispondere a come sono stati generati i CSV forniti, serve solo "
                          "per calcolare il bracket da evidenziare (vedi docstring)")
    ap.add_argument("--out", default="results/dp_low_p_scaling_n.png",
                     help="path base del file immagine: un file per m, vedi docstring del modulo")
    args = ap.parse_args()

    df = load_scaling_sweep_results(args.scaling_csv, args.sweep_csv)
    m_values = args.m if args.m is not None else sorted(df["m"].unique())
    plot_scaling_sweep(df, m_values=m_values, representation_mode=args.representation, out_path=args.out)


if __name__ == "__main__":
    main()
