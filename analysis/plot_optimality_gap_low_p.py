"""
Confronto della QUALITA' delle soluzioni trovate dai tre metodi (MILP, DP,
LPT) -- stessa identica logica di plot_optimality_gap.py (classificazione
ammissibile/non-ammissibile della DP in OOM, evidenziazione "OOM k/N",
etichette di differenza percentuale rispetto a MILP, vedi quel modulo per
il dettaglio completo), con UNA differenza: quello script assume 'variance'
in {'low', 'high'} (le due classi fisse di experiment_runner.py), qui
invece i CSV vengono da experiments/run_low_p_experiment.py, dove
'variance' e' la stringa descrittiva prodotta da
generate_instance_custom_range() -- tipicamente "custom[p_low,p_high]"
(es. "custom[1,5]"), diversa run per run, quindi non hardcodabile in
_VARIANCE_LABELS come nello script gemello.

Le uniche differenze rispetto a plot_optimality_gap.py sono:
  - _variance_label()/_variance_sort_key() al posto di _VARIANCE_LABELS:
    riconoscono il pattern "custom[a,b]" e lo rendono leggibile come
    "p in [a,b]" nel titolo del grafico (invece della stringa grezza --
    plot_optimality_gap.py non andava comunque in errore su un valore non
    riconosciuto, usa gia' un .get(variance, variance) di fallback, ma
    stampava la stringa grezza "custom[1,5]" cosi' com'e', meno leggibile);
  - _variance_slug() per il nome file (es. "p1-5" invece di "custom[1,5]":
    parentesi quadre e virgola sono scomode da referenziare da shell,
    "[...]" e' un pattern di glob);
  - --variance nella CLI accetta qualunque stringa, non solo low/high;
  - viene creata la cartella di --out se non esiste gia' (in
    plot_optimality_gap.py mancava, fig.savefig() falliva con
    FileNotFoundError se la cartella non era gia' presente -- stesso bug
    gia' trovato e corretto nell'adattamento di plot_solver_time_by_n.py);
  - classify_dp_rows() riconosce un TERZO pattern di dp_note, oltre ai due
    di plot_optimality_gap.py ("incumbent ammissibile" / "non e' un cmax
    ammissibile"): "nessun incumbent disponibile", emesso quando la DP va
    OOM sulla rappresentazione DENSA invece che sparsa -- la densa alloca
    l'intero array in un colpo solo (np.zeros(shape)), quindi se
    quell'allocazione fallisce non c'e' nessun progresso job-per-job da
    riportare (a differenza della sparsa, che processa un job alla volta
    e puo' riportare il miglior stato raggiunto finora) e dp_cmax e' NaN,
    non un incumbent parziale. Trattato come non ammissibile, stesso
    bucket di "non e' un cmax ammissibile" (zero informazione su un cmax
    dell'istanza intera, solo per un motivo diverso: nessun dato affatto
    invece di un dato parziale non completabile). Senza questo pattern,
    classify_dp_rows() lo classificherebbe come "non riconosciuto" e lo
    escluderebbe con un avviso invece di trattarlo correttamente.
Il resto -- la logica di aggregazione, l'evidenziazione OOM parziale/
totale, le etichette di differenza percentuale -- e' identico, copiato
senza modifiche.

Uso da linea di comando:

    python analysis/plot_optimality_gap_low_p.py \
        --csv results/results_low_p.csv \
        --out results/optimality_value_low_p

produce, per ciascuna coppia (m, variance) presente nei dati, ad es.:
    results/optimality_value_low_p_m3_p1-5.png, ...

Oppure programmaticamente:

    from analysis.plot_optimality_gap_low_p import (load_optimality_results,
                                                      classify_dp_rows,
                                                      plot_optimality_values)
    df = load_optimality_results(["results/results_low_p.csv"])
    plot_optimality_values(df, out_path="results/optimality_value_low_p")
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance", "milp_cmax", "lpt_cmax",
                          "dp_cmax", "dp_status", "dp_is_optimal"}

# identici a plot_optimality_gap.py, vedi quel modulo per il significato
_NOTE_NON_ADMISSIBLE = "non e' un cmax ammissibile"
_NOTE_ADMISSIBLE_FALLBACK = "incumbent ammissibile"
# terzo pattern, distinto dagli altri due: OOM sulla rappresentazione DENSA,
# che alloca l'intero array in un colpo solo (np.zeros(shape)) -- se fallisce
# li' non c'e' nessun progresso job-per-job da riportare (a differenza della
# sparsa, che processa un job alla volta), quindi dp_cmax e' NaN, non un
# incumbent parziale. Non ammissibile per lo stesso motivo della sparsa (zero
# informazione su un cmax dell'istanza intera), solo per una ragione diversa
# (nessun dato affatto, invece di un dato parziale non completabile).
_NOTE_NO_INCUMBENT = "nessun incumbent disponibile"

_METHOD_COLORS = {"MILP": "tab:blue", "DP": "tab:orange", "LPT": "tab:green"}
_METHOD_ORDER = ["MILP", "DP", "LPT"]

_OOM_HATCH = "//"
_OOM_EDGE = "firebrick"
_BAR_GAP = 0.08

_CUSTOM_RANGE_RE = re.compile(r"^custom\[(\d+),\s*(\d+)\]$")
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
    (parentesi quadre/virgola sono scomode da shell, "[...]" e' un pattern
    di glob) -- l'etichetta leggibile nel titolo resta _variance_label()."""
    m = _CUSTOM_RANGE_RE.match(str(variance))
    if m:
        return f"p{m.group(1)}-{m.group(2)}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(variance))


def load_optimality_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Identica a load_optimality_results di plot_optimality_gap.py."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_optimality_gap_low_p] rimosse {n_before - len(df)} righe "
                  f"duplicate (stesso instance_id in piu' file)")

    return df.sort_values(["m", "n", "variance"]).reset_index(drop=True)


def classify_dp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Identica a classify_dp_rows di plot_optimality_gap.py, vedi quel
    modulo per il dettaglio: aggiunge 'dp_admissible' (bool/NA)."""
    df = df.copy()
    note = df.get("dp_note", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()

    is_optimal = df["dp_is_optimal"] == True  # noqa: E712
    is_admissible_fallback = note.str.contains(_NOTE_ADMISSIBLE_FALLBACK, regex=False)
    is_non_admissible = (note.str.contains(_NOTE_NON_ADMISSIBLE, regex=False)
                          | note.str.contains(_NOTE_NO_INCUMBENT, regex=False))

    admissible = pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    admissible[is_optimal] = True
    admissible[~is_optimal & is_admissible_fallback] = True
    admissible[~is_optimal & is_non_admissible] = False

    unrecognized = admissible.isna() & (df["dp_status"] == "MEMORY_LIMIT_EXCEEDED")
    n_unrecognized = int(unrecognized.sum())
    if n_unrecognized:
        print(f"[plot_optimality_gap_low_p] ATTENZIONE: {n_unrecognized} righe con "
              f"dp_status=MEMORY_LIMIT_EXCEEDED ma dp_note non riconosciuta "
              f"(ne' '{_NOTE_ADMISSIBLE_FALLBACK}' ne' '{_NOTE_NON_ADMISSIBLE}' ne' "
              f"'{_NOTE_NO_INCUMBENT}'): escluse dai grafici invece di essere classificate "
              f"a caso. Controlla dp_note per queste righe.")

    df["dp_admissible"] = admissible
    return df


def _aggregate(values: pd.Series, agg: str) -> Optional[float]:
    values = values.dropna()
    if values.empty:
        return None
    return float(values.median() if agg == "median" else values.mean())


def plot_optimality_values(df: pd.DataFrame, m_values: Optional[List[int]] = None,
                            variance_values: Optional[List[str]] = None, agg: str = "median",
                            diff_labels: bool = True,
                            out_path: str = "results/optimality_value_low_p.png") -> None:
    """Identica a plot_optimality_values di plot_optimality_gap.py, salvo
    l'ordinamento delle coppie (m, variance) con _variance_sort_key() al
    posto di un ordinamento alfabetico puro."""
    if agg not in ("median", "mean"):
        raise ValueError(f"agg deve essere 'median' o 'mean' (ricevuto: {agg!r})")
    if "dp_admissible" not in df.columns:
        df = classify_dp_rows(df)

    sub = df
    if m_values is not None:
        sub = sub[sub["m"].isin(m_values)]
    if variance_values is not None:
        sub = sub[sub["variance"].isin(variance_values)]
    if sub.empty:
        raise ValueError("nessuna riga da plottare dopo il filtro")

    root, ext = os.path.splitext(out_path)
    pairs = sub[["m", "variance"]].drop_duplicates()
    pairs["_variance_order"] = pairs["variance"].map(lambda v: _variance_sort_key(v)[0])
    pairs = pairs.sort_values(["m", "_variance_order", "variance"]).itertuples(index=False)
    for m, variance, _ in pairs:
        pair_sub = sub[(sub["m"] == m) & (sub["variance"] == variance)]
        _plot_single_pair(pair_sub, m=m, variance=variance, agg=agg, diff_labels=diff_labels,
                           out_path=f"{root}_m{m}_{_variance_slug(variance)}{ext}")


def _plot_single_pair(sub: pd.DataFrame, m: int, variance: str, agg: str, diff_labels: bool,
                       out_path: str) -> None:
    n_list = sorted(sub["n"].unique())
    if not n_list:
        print(f"[plot_optimality_gap_low_p] nessun valore di n per m={m}, "
              f"variance={variance}, salto")
        return

    n_methods = len(_METHOD_ORDER)
    bar_width = (1.0 - _BAR_GAP) / n_methods
    group_positions = {n_val: i for i, n_val in enumerate(n_list)}
    offsets = {name: (j - (n_methods - 1) / 2) * bar_width for j, name in enumerate(_METHOD_ORDER)}

    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(n_list) * n_methods), 6.0))

    any_oom = False
    for n_val in n_list:
        rows = sub[sub["n"] == n_val]
        pos_milp = group_positions[n_val] + offsets["MILP"]
        pos_dp = group_positions[n_val] + offsets["DP"]
        pos_lpt = group_positions[n_val] + offsets["LPT"]

        milp_h = _aggregate(rows["milp_cmax"], agg)
        lpt_h = _aggregate(rows["lpt_cmax"], agg)

        if milp_h is not None:
            ax.bar(pos_milp, milp_h, width=bar_width * 0.92, color=_METHOD_COLORS["MILP"],
                   label="_nolegend_")

        n_total = len(rows)
        n_admissible = int(rows["dp_admissible"].fillna(False).sum())
        n_failed = int((rows["dp_status"] == "MEMORY_LIMIT_EXCEEDED").sum())

        if n_admissible == 0:
            any_oom = True
            if milp_h is not None:
                ax.bar(pos_dp, milp_h, width=bar_width * 0.92, facecolor="none",
                       edgecolor=_OOM_EDGE, hatch=_OOM_HATCH, linewidth=1.3, label="_nolegend_")
                ax.text(pos_dp, milp_h * 1.02, f"OOM ({n_failed}/{n_total})", ha="center",
                        va="bottom", fontsize=7.5, color=_OOM_EDGE, fontweight="bold")
        else:
            dp_h = _aggregate(rows.loc[rows["dp_admissible"].fillna(False), "dp_cmax"], agg)
            if dp_h is not None:
                if n_failed > 0:
                    any_oom = True
                    ax.bar(pos_dp, dp_h, width=bar_width * 0.92, color=_METHOD_COLORS["DP"],
                           edgecolor=_OOM_EDGE, hatch=_OOM_HATCH, linewidth=1.3, label="_nolegend_")
                    ax.text(pos_dp, dp_h * 1.02, f"OOM ({n_failed}/{n_total})", ha="center",
                            va="bottom", fontsize=7.5, color=_OOM_EDGE, fontweight="bold")
                else:
                    ax.bar(pos_dp, dp_h, width=bar_width * 0.92, color=_METHOD_COLORS["DP"],
                           label="_nolegend_")
                    if diff_labels and milp_h and milp_h != 0:
                        pct = 100.0 * (dp_h - milp_h) / milp_h
                        ax.text(pos_dp, dp_h * 1.02, f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%",
                                ha="center", va="bottom", fontsize=7.5, color="dimgray")

        if lpt_h is not None:
            ax.bar(pos_lpt, lpt_h, width=bar_width * 0.92, color=_METHOD_COLORS["LPT"],
                   label="_nolegend_")
            if diff_labels and milp_h and milp_h != 0:
                pct = 100.0 * (lpt_h - milp_h) / milp_h
                ax.text(pos_lpt, lpt_h * 1.02, f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=7.5, color="dimgray")

    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels([str(n_val) for n_val in n_list])
    ax.set_xlabel("numero di job (n)")
    ax.set_ylabel(f"cmax ({agg} sulle istanze della classe)")
    ax.set_title(f"MILP vs DP vs LPT -- m={m}, {_variance_label(variance)}")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    if any_oom:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.15 * (top - bottom))
    else:
        ax.margins(y=0.12)

    handles = [Patch(facecolor=_METHOD_COLORS[name], label=name) for name in _METHOD_ORDER]
    if any_oom:
        handles.append(Patch(facecolor="white", edgecolor=_OOM_EDGE, hatch=_OOM_HATCH,
                              label="OOM"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_optimality_gap_low_p] grafico salvato in {out_path}")


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
                              "nei CSV; a differenza di plot_optimality_gap.py non e' ristretto "
                              "a low/high, vedi docstring del modulo)")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra piu' istanze della stessa classe (m, n, variance)")
    parser.add_argument("--no-diff-labels", dest="diff_labels", action="store_false", default=True,
                         help="non mostrare l'etichetta con la differenza percentuale sopra le colonne")
    parser.add_argument("--out", default="results/optimality_value_low_p.png",
                         help="path base del file immagine di output: un file per ciascuna "
                              "combinazione (m, variance), vedi docstring del modulo")
    args = parser.parse_args()

    df = load_optimality_results(args.csv)
    df = classify_dp_rows(df)

    plot_optimality_values(df, m_values=args.m, variance_values=args.variance, agg=args.agg,
                            diff_labels=args.diff_labels, out_path=args.out)


if __name__ == "__main__":
    main()
