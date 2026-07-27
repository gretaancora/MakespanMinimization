"""
Andamento del tempo della DP al variare di n, un grafico a barre distinto
per ciascun valore di m -- pensato per i CSV prodotti da calibrate_dp_lpt.py
(servono le colonne 'm', 'variance', 'n', 'dp_status', 'dp_runtime'; se
presente, la colonna 'representation' viene usata per isolare una singola
rappresentazione: esplicitamente con --representation sparse|dense,
oppure, per default (--representation non passato), tenendo per ciascuna
riga solo quella la cui 'representation' coincide con la colonna
'auto_pick' del CSV, cioe' la rappresentazione che choose_representation
avrebbe scelto automaticamente per QUELLA specifica istanza (m, variance,
n) -- vedi --representation piu' sotto).

Per ogni m viene prodotto un grafico a barre con un gruppo di barre per
ciascun valore di n presente nel grid di quel m: dentro al gruppo, una barra
per 'variance=low' e una per 'variance=high', ma SOLO se la misurazione e'
disponibile per quella combinazione (n, variance) -- se per un certo n e'
stata misurata solo la varianza bassa (o solo l'alta), il gruppo ha una sola
barra invece di due, non una barra "vuota" al posto di quella mancante.

Le combinazioni (n, variance) per cui la DP e' andata realmente in
out-of-memory (dp_status == "MEMORY_LIMIT_EXCEEDED") sono evidenziate con lo
stesso schema usato per l'OOM in plot_optimality_gap.py: il colore di
riempimento della barra resta quello della variance, ma il bordo diventa
firebrick e viene aggiunto un tratteggio "//", cosi' da distinguerle a
colpo d'occhio dalle barre "OPTIMAL" anche quando l'altezza (dp_runtime) e'
comparabile. Da non confondere con TIME_SOFT_CAP_STOPPED (vedi
calibrate_dp_lpt.py): quello NON e' un fallimento della DP, e' solo il punto
in cui lo script di calibrazione ha smesso di scalare n per non far durare
la calibrazione all'infinito -- qui non viene evidenziato separatamente,
resta leggibile dal fatto che la barra e' vicina o sopra la linea del soft
cap.

Una linea orizzontale tratteggiata segna il soft cap di tempo (--time-soft-
cap-s, default 120s per allinearsi al default di calibrate_dp_lpt.py): serve
solo da riferimento visivo, non filtra ne' altera i dati.

L'asse y e' in scala log: dp_runtime copre diversi ordini di grandezza (da
pochi millisecondi a centinaia di secondi) anche all'interno dello stesso m,
quindi una scala lineare schiaccerebbe le barre piu' basse.

Uso da linea di comando:

    python analysis/plot_dp_calibration_bars.py \
        --csv results/calibration_dp_representation.csv \
        --time-soft-cap-s 120 --out results/dp_calibration_runtime.png

produce un file per ciascun m presente nel CSV, ad es.
results/dp_calibration_runtime_m2.png, ..._m3.png, ..._m5.png.

Oppure programmaticamente:

    from analysis.plot_dp_calibration_bars import (load_dp_calibration_results,
                                                     plot_dp_runtime_bars)
    df = load_dp_calibration_results(["results/calibration_dp_representation.csv"])
    plot_dp_runtime_bars(df, m_values=[2, 3, 5], time_soft_cap=120.0,
                          out_path="results/dp_calibration_runtime.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED_COLUMNS = {"m", "variance", "n", "dp_status", "dp_solver_runtime"}

# palette fissa per variance, cosi' lo stesso valore ha sempre lo stesso
# colore anche confrontando grafici prodotti in run diverse (stesso
# principio della palette per m in plot_milp_scaling.py)
_VARIANCE_COLORS = {"low": "tab:blue", "high": "tab:orange"}
# stesso schema di evidenziazione OOM usato in plot_optimality_gap.py: il
# colore di riempimento resta quello "normale" (qui quello della variance),
# l'evidenziazione e' data dal bordo e dal tratteggio, non da un colore
# di riempimento diverso
_OOM_HATCH = "//"
_OOM_EDGE = "firebrick"
_BAR_WIDTH = 0.35


def load_dp_calibration_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Concatena i CSV di calibrazione DP indicati e rimuove eventuali righe
    duplicate (stessa combinazione m/n/variance, e representation se la
    colonna e' presente) nel caso gli stessi punti compaiano in piu' file."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    key_cols = [c for c in ["m", "n", "variance", "representation"] if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    if len(df) < n_before:
        print(f"[plot_dp_calibration_bars] rimosse {n_before - len(df)} righe duplicate "
              f"(stessa combinazione {key_cols} in piu' file)")

    return df.sort_values(["m", "n", "variance"]).reset_index(drop=True)


def plot_dp_runtime_bars(df: pd.DataFrame, m_values: List[int],
                          representation: Optional[str] = None,
                          time_soft_cap: Optional[float] = 120.0,
                          out_path: str = "results/dp_calibration_runtime.png") -> None:
    """Un file immagine per ciascun m in m_values (out_path e' usato come
    base: 'results/dp_calibration_runtime.png' diventa
    'results/dp_calibration_runtime_m2.png', '..._m3.png', ecc., stesso
    principio di out_path in plot_milp_scaling_by_variance).

    representation: se la colonna 'representation' e' presente nel CSV e ci
    sono piu' rappresentazioni diverse per la stessa combinazione (m, n,
    variance) -- tipico di una calibrazione lanciata con
    --representation both in calibrate_dp_lpt.py -- va isolata quella da
    plottare, altrimenti la stessa barra avrebbe piu' di un valore possibile
    di dp_runtime. Due modalita':
      - representation="sparse" oppure "dense": tiene solo le righe con
        quella rappresentazione esplicita.
      - representation=None (default): tiene, per ciascuna riga, solo
        quella la cui 'representation' coincide con la colonna 'auto_pick'
        del CSV, cioe' la rappresentazione che choose_representation
        avrebbe scelto automaticamente per QUELLA specifica istanza (m,
        variance, n) -- vedi calibrate_dp_lpt.py. Utile per vedere il
        runtime che si otterrebbe davvero in campagna, dove la scelta
        della rappresentazione e' automatica e puo' cambiare al variare di
        n anche dentro lo stesso (m, variance).
    Se la colonna 'representation' non e' presente nel CSV (calibrazione a
    singola rappresentazione), il parametro viene ignorato in entrambi i
    casi. Se representation=None e la colonna 'auto_pick' non e' presente,
    viene sollevato un errore esplicito, dato che senza quella colonna la
    modalita' 'auto' non e' applicabile."""
    if "representation" in df.columns:
        if representation is not None:
            available = sorted(df["representation"].dropna().unique())
            df = df[df["representation"] == representation]
            if df.empty:
                raise ValueError(f"nessuna riga con representation='{representation}' "
                                  f"(disponibili nel CSV: {available})")
        else:
            if "auto_pick" not in df.columns:
                raise ValueError(
                    "representation=None (modalita' auto) richiede la colonna 'auto_pick' "
                    "nel CSV (prodotta da calibrate_dp_lpt.py) per isolare, per ciascuna "
                    "istanza, la rappresentazione scelta automaticamente da "
                    "choose_representation; specificare esplicitamente representation="
                    "'sparse'|'dense' oppure fornire un CSV che contenga 'auto_pick'")
            df = df[df["representation"] == df["auto_pick"]]
            if df.empty:
                raise ValueError("nessuna riga con representation == auto_pick nel CSV fornito "
                                  "(modalita' auto)")

    for m in sorted(m_values):
        sub = df[df["m"] == m]
        if sub.empty:
            print(f"[plot_dp_calibration_bars] nessuna riga per m={m}, salto")
            continue
        root, ext = os.path.splitext(out_path)
        _plot_single_m(sub, m=m, time_soft_cap=time_soft_cap, out_path=f"{root}_m{m}{ext}")


def _plot_single_m(sub: pd.DataFrame, m: int, time_soft_cap: Optional[float],
                    out_path: str) -> None:
    n_values = sorted(sub["n"].unique())
    positions_by_n = {n: i for i, n in enumerate(n_values)}
    offsets = {"low": -_BAR_WIDTH / 2, "high": _BAR_WIDTH / 2}

    fig, ax = plt.subplots(figsize=(max(8.0, len(n_values) * 0.9), 6.5))

    oom_points = []  # (n, variance, dp_peak_memory_mb) per il riepilogo testuale
    plotted_any_bar = False

    for variance in ("low", "high"):
        rows = sub[sub["variance"] == variance]
        if rows.empty:
            continue

        positions, heights, colors, edgecolors, hatches = [], [], [], [], []
        for _, row in rows.sort_values("n").iterrows():
            is_oom = row["dp_status"] == "MEMORY_LIMIT_EXCEEDED"
            positions.append(positions_by_n[row["n"]] + offsets[variance])
            heights.append(row["dp_solver_runtime"])
            colors.append(_VARIANCE_COLORS[variance])
            edgecolors.append(_OOM_EDGE if is_oom else "none")
            hatches.append(_OOM_HATCH if is_oom else None)
            if is_oom:
                oom_points.append((row["n"], variance, row.get("dp_peak_memory_mb")))

        bars = ax.bar(positions, heights, width=_BAR_WIDTH, color=colors,
                       edgecolor=edgecolors, hatch=hatches, linewidth=1.3,
                       label=f"variance={variance}")
        plotted_any_bar = True

    if not plotted_any_bar:
        plt.close(fig)
        print(f"[plot_dp_calibration_bars] nessuna barra da disegnare per m={m}, salto")
        return

    if time_soft_cap is not None:
        ax.axhline(time_soft_cap, color="gray", linestyle="--", linewidth=1.2,
                   label=f"soft cap ({time_soft_cap:.0f}s)")

    ax.set_xticks(list(positions_by_n.values()))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yscale("log")
    ax.set_xlabel("numero di job")
    ax.set_ylabel("dp_runtime (s, scala log)")
    ax.set_title(f"Tempo della DP al variare di n (m={m})")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if oom_points:
        handles.append(Patch(facecolor="white", edgecolor=_OOM_EDGE, hatch=_OOM_HATCH))
        labels.append("out of memory")
    ax.legend(handles, labels, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_dp_calibration_bars] grafico salvato in {out_path}")

    if oom_points:
        print(f"[plot_dp_calibration_bars] combinazioni (n, variance) in out-of-memory "
              f"per m={m}:")
        for n, variance, mem in oom_points:
            mem_str = f"{mem:.1f}MB" if mem is not None else "n/d"
            print(f"  n={n:>4}  variance={variance:<5}  dp_peak_memory_mb={mem_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di calibrazione DP da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=None,
                         help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--representation", choices=["sparse", "dense"], default=None,
                         help="rappresentazione da isolare esplicitamente se la colonna "
                              "'representation' e' presente e contiene piu' valori per la "
                              "stessa combinazione (m, n, variance). Default: non passato, cioe' "
                              "per ciascuna riga viene tenuta solo la rappresentazione che "
                              "coincide con la colonna 'auto_pick' del CSV (quella scelta "
                              "automaticamente da choose_representation per quella specifica "
                              "istanza) -- ignorato se la colonna 'representation' non e' "
                              "presente nel CSV")
    parser.add_argument("--time-soft-cap-s", type=float, default=120.0,
                         help="soft cap di tempo di calibrate_dp_lpt.py, per la linea di "
                              "riferimento (default: 120; usare --time-soft-cap-s -1 per non "
                              "disegnarla)")
    parser.add_argument("--out", default="results/dp_calibration_runtime.png",
                         help="path base del file immagine di output: un file per m, vedi "
                              "docstring del modulo")
    args = parser.parse_args()

    df = load_dp_calibration_results(args.csv)
    m_values = args.m if args.m is not None else sorted(df["m"].unique())
    time_soft_cap = None if args.time_soft_cap_s < 0 else args.time_soft_cap_s

    plot_dp_runtime_bars(df, m_values=m_values, representation=args.representation,
                         time_soft_cap=time_soft_cap, out_path=args.out)


if __name__ == "__main__":
    main()
