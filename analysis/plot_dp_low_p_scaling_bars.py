"""
Grafico a barre del tempo di risoluzione della DP al variare di n, un file
per ciascun m -- una barra per n (non una per istanza/seed): l'altezza e'
la mediana (default, vedi --agg) o la media di dp_runtime sulle istanze
misurate a quell'n. La mediana e' il default perche' dp_runtime tra istanze
diverse allo stesso n puo' variare parecchio (anche 3-4x, seed diversi
producono istanze diverse pur restando in [p_low, p_high] fisso) ed e' piu'
robusta a un singolo run anomalo di quanto lo sia la media; --agg mean per
tornare alla media se preferita.

L'aggregazione usa TUTTE le istanze misurate a quell'n, comprese quelle
finite in MEMORY_LIMIT_EXCEEDED (per cui dp_runtime e' il tempo trascorso
PRIMA del crash, non un tempo-a-soluzione, vedi calibrate_dp_low_p_scaling_n.py):
mescolarle con le istanze OPTIMAL non e' concettualmente pulitissimo, ma e'
comunque un'indicazione ragionevole di "quanto ci mette la DP a quell'n"
prima di sapere se chiudera' o no, che e' l'uso pratico di questo grafico.
Il conteggio delle istanze OOM a quell'n resta comunque visibile per intero
(vedi sotto), cosi' non e' un'informazione persa, solo non scorporata
dall'altezza della barra.

Le barre sono di un unico colore (arancione): qui non c'e' evidenziazione
per bordo/tratteggio dei singoli esiti (a differenza di
plot_dp_calibration_bars.py e delle versioni precedenti di questo script)
-- l'informazione su OOM e soft cap e' data altrove, non sullo stile della
barra:
  - se almeno una istanza a quell'n e' MEMORY_LIMIT_EXCEEDED, sopra la
    barra compare l'etichetta "OOM(x/N)" (x = istanze MEMORY_LIMIT_EXCEEDED,
    N = istanze totali misurate a quell'n -- N puo' variare da un n
    all'altro se scaling e sweep hanno usato --instances-per-class diversi,
    quindi non e' mai dato per scontato che sia 3);
  - il soft cap di tempo (--time-soft-cap-s, default 120s) resta SOLO la
    linea orizzontale tratteggiata di riferimento, senza nessuna
    evidenziazione sulle barre stesse.

L'asse y e' in scala log (dp_runtime copre facilmente 4-5 ordini di
grandezza tra i punti piu' piccoli e quelli ridosso del fallimento).

Il grafico si ferma al primo n in cui TUTTE le istanze misurate sono
MEMORY_LIMIT_EXCEEDED (OOM totale): gli n successivi, se presenti nel CSV
(es. lo scaling ne aggiunge sempre uno in piu' per verificare che il
fallimento sia sistematico, vedi calibrate_dp_low_p_scaling_n.py), sono
comunque oltre il punto in cui e' gia' out-of-memory su tutta la classe e
vengono scartati dal disegno (con un avviso a console) invece di allungare
inutilmente l'asse x.

Uso:
    python analysis/plot_dp_low_p_scaling_bars.py \
        --scaling-csv results/calibration_dp_low_p_scaling_n.csv \
        --sweep-csv results/calibration_dp_low_p_sweep_n.csv \
        --agg median --time-soft-cap-s 120 --out results/dp_low_p_scaling_bars.png
"""
from __future__ import annotations
import argparse
import os
import statistics
import sys
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_dp_low_p_scaling_n import load_scaling_sweep_results, _nan_to_none  # riuso diretto

_BAR_COLOR = "tab:orange"
_BAR_WIDTH = 0.7
_OOM_TEXT_COLOR = "firebrick"


def plot_dp_runtime_bars(df: pd.DataFrame, m_values: List[int], agg: str = "median",
                          time_soft_cap: Optional[float] = 120.0,
                          out_path: str = "results/dp_low_p_scaling_bars.png") -> None:
    for m in sorted(m_values):
        sub = df[df["m"] == m]
        if sub.empty:
            print(f"[plot_dp_low_p_scaling_bars] nessuna riga per m={m}, salto")
            continue
        root, ext = os.path.splitext(out_path)
        _plot_single_m(sub, m=m, agg=agg, time_soft_cap=time_soft_cap, out_path=f"{root}_m{m}{ext}")


def _plot_single_m(sub: pd.DataFrame, m: int, agg: str, time_soft_cap: Optional[float],
                    out_path: str) -> None:
    rows = _nan_to_none(sub.to_dict("records"))
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    ns_all = sorted(by_n.keys())

    # ferma il grafico al primo n con OOM totale (tutte le istanze misurate a
    # quell'n sono MEMORY_LIMIT_EXCEEDED): gli n successivi sono comunque
    # oltre il punto in cui e' gia' out-of-memory su tutta la classe, non
    # aggiungono informazione e allungano solo l'asse x
    ns = []
    for n in ns_all:
        ns.append(n)
        rs = by_n[n]
        n_oom = sum(1 for r in rs if r["dp_status"] == "MEMORY_LIMIT_EXCEEDED")
        if n_oom == len(rs):
            if len(ns) < len(ns_all):
                print(f"[plot_dp_low_p_scaling_bars] m={m}: OOM totale gia' raggiunto a n={n}, "
                      f"non disegno i {len(ns_all) - len(ns)} n successivi ({ns_all[len(ns):]})")
            break

    positions_by_n = {n: i for i, n in enumerate(ns)}
    agg_fn = statistics.median if agg == "median" else statistics.mean

    fig, ax = plt.subplots(figsize=(max(9.0, len(ns) * 0.7), 6.0))

    for n in ns:
        rs = by_n[n]
        height = agg_fn(r["dp_runtime"] for r in rs)
        ax.bar(positions_by_n[n], height, width=_BAR_WIDTH, color=_BAR_COLOR)

        n_oom = sum(1 for r in rs if r["dp_status"] == "MEMORY_LIMIT_EXCEEDED")
        if n_oom > 0:
            ax.text(positions_by_n[n], height * 1.12, f"OOM({n_oom}/{len(rs)})", ha="center",
                    va="bottom", fontsize=7.5, color=_OOM_TEXT_COLOR, fontweight="bold")

    if time_soft_cap is not None:
        ax.axhline(time_soft_cap, color="gray", linestyle="--", linewidth=1.2,
                   label=f"soft cap ({time_soft_cap:.0f}s)")
        ax.legend(loc="upper left", fontsize=9)

    ax.set_xticks(list(positions_by_n.values()))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yscale("log")
    ax.set_xlabel("numero di job")
    ax.set_ylabel(f"dp_runtime (s, scala log)")
    ax.set_title(f"Tempo della DP al variare di n (m={m})")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_low_p_scaling_bars] salvato {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling-csv", required=True,
                     help="CSV di dettaglio di calibrate_dp_low_p_scaling_n.py")
    ap.add_argument("--sweep-csv", nargs="*", default=[],
                     help="uno o piu' CSV di dettaglio di calibrate_dp_low_p_sweep_n.py (opzionale)")
    ap.add_argument("--m", nargs="+", type=int, default=None,
                     help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    ap.add_argument("--agg", choices=["median", "mean"], default="median",
                     help="statistica usata per aggregare dp_runtime sulle istanze allo stesso n "
                          "(default: median, piu' robusta a run anomali, vedi docstring)")
    ap.add_argument("--time-soft-cap-s", type=float, default=120.0,
                     help="soft cap di tempo, per la linea di riferimento (default: 120; usare "
                          "--time-soft-cap-s -1 per non disegnarla)")
    ap.add_argument("--out", default="results/dp_low_p_scaling_bars.png",
                     help="path base del file immagine: un file per m, vedi docstring del modulo")
    args = ap.parse_args()

    df = load_scaling_sweep_results(args.scaling_csv, args.sweep_csv)
    m_values = args.m if args.m is not None else sorted(df["m"].unique())
    time_soft_cap = None if args.time_soft_cap_s < 0 else args.time_soft_cap_s

    plot_dp_runtime_bars(df, m_values=m_values, agg=args.agg, time_soft_cap=time_soft_cap,
                         out_path=args.out)


if __name__ == "__main__":
    main()
