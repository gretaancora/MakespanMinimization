"""
Confronto della QUALITA' delle soluzioni trovate dai tre metodi (MILP,
DP, LPT) -- pensato per i CSV prodotti da experiment_runner.py (servono,
oltre a 'm'/'n'/'variance', le colonne 'milp_cmax', 'lpt_cmax', 'dp_cmax',
'dp_status', 'dp_is_optimal', 'dp_note').

Rispetto a una prima versione di questo modulo (due grafici separati,
MILP-vs-LPT e MILP-vs-DP) e a una intermedia (due grafici, uno per
coppia), qui i TRE metodi condividono un unico grafico: sull'asse x i
valori di n, per ciascuno un gruppo di TRE colonne affiancate (MILP, DP,
LPT, stesso ordine/palette usati in plot_solver_time_by_n.py e
plot_solver_time_by_variance.py, cosi' i colori sono consistenti in tutta
la cartella analysis/). Un file per ciascuna coppia (m, variance) -- si
resta su questa scelta della versione precedente: il cmax tipico dipende
moltissimo dalla varianza dei tempi di lavorazione (istanze "high
variance" hanno cmax anche 10 volte piu' grandi di quelle "low variance"
a parita' di m, n), mischiare le due varianze nello stesso grafico
schiaccerebbe la differenza che si vuole mostrare.

--------------------------------------------------------------------------
PERCHE' LA DP HA BISOGNO DI UN TRATTAMENTO A PARTE (OOM)
--------------------------------------------------------------------------
MILP e LPT restituiscono sempre una soluzione ammissibile per l'istanza
intera (MILP e' sempre OPTIMAL nei CSV disponibili; LPT e' un'euristica
O(n log n) che non fallisce mai). La DP no: quando esaurisce la memoria
(dp_status == "MEMORY_LIMIT_EXCEEDED"), a seconda della versione del
solver usata per produrre il CSV, dp_cmax puo' essere:
  (a) un cmax ammissibile ma sub-ottimo (vecchia logica con
      completamento greedy LPT sui job restanti, dp_note contiene
      "incumbent ammissibile"), oppure
  (b) il cmax del miglior stato raggiunto sui SOLI primi n_jobs_dp_exact
      job, che NON e' una soluzione per l'istanza intera (dp_note
      contiene "non e' un cmax ammissibile", logica attuale di
      solvers/dp_solver.py): un numero non confrontabile col cmax
      dell'istanza completa.
classify_dp_rows() distingue le due situazioni leggendo dp_is_optimal
insieme al testo di dp_note (non il segno/valore del gap, un sintomo
indiretto e quindi fragile). Questa classificazione decide ANCORA quali
righe entrano nella media/mediana della colonna DP (non ha senso
mediare un dp_cmax non ammissibile insieme a uno ammissibile: non sono
la stessa grandezza). L'etichetta numerica "OOM x/N", pero', NON conta
piu' le non-ammissibili: conta le istanze con dp_status ==
"MEMORY_LIMIT_EXCEEDED" cosi' come sono, un fallimento e' un fallimento
a prescindere dal fatto che il suo incumbent sia stato classificato
ammissibile o meno. plot_optimality_values() usa questa classificazione
per EVIDENZIARE la colonna DP invece di disegnarla come se fosse un
valore normale:
  - tutte le istanze della classe (m, n, variance) ammissibili (ottime o
    sub-ottime): colonna normale, nessuna evidenziazione, con l'etichetta
    della differenza percentuale rispetto a MILP (come per LPT);
  - almeno una istanza non ammissibile (OOM), ma non tutte: colonna
    tratteggiata, calcolata SOLO sulle istanze ammissibili, con
    etichetta "OOM k/N" (k = istanze con dp_status ==
    "MEMORY_LIMIT_EXCEEDED" nella classe, N = totale) al posto della
    differenza percentuale;
  - tutte le istanze non ammissibili (OOM totale): nessun dp_cmax
    confrontabile esiste per quella classe -- la colonna non viene
    disegnata con un valore inventato, ma sostituita da un marcatore
    tratteggiato all'altezza del MILP con l'etichetta "OOM (N/N)", cosi'
    il buco e' visibile invece di sembrare un dato dimenticato (PRIMA di
    questa modifica il numeratore qui era ammissibili, quindi sempre 0:
    fuorviante, dava l'impressione di "0 fallite" anziche' "tutte
    fallite" -- vedi nota su n_failed in _plot_single_pair).
Se dp_note non corrisponde a nessuno dei due pattern noti, la riga viene
scartata dalla media/mediana con un avviso a schermo invece di essere
classificata a indovinare (ma resta comunque contata in k se il suo
dp_status e' "MEMORY_LIMIT_EXCEEDED": l'etichetta conta i fallimenti,
non dipende dal riconoscimento della nota).

LPT non ha mai questo problema (euristica in-process, sempre ammissibile),
quindi la sua colonna mostra sempre l'etichetta di differenza percentuale
rispetto a MILP, indipendentemente da cosa succede alla DP nello stesso
gruppo.

Uso da linea di comando:

    python analysis/plot_optimality_gap.py \\
        --csv results/experiment1.csv results/experiment2.csv \\
        --out results/optimality_value

produce, per ciascuna coppia (m, variance) presente nei dati:
    results/optimality_value_m2_low.png, results/optimality_value_m2_high.png, ...

Oppure programmaticamente:

    from analysis.plot_optimality_gap import (load_optimality_results,
                                                classify_dp_rows,
                                                plot_optimality_values)
    df = load_optimality_results(["results/experiment1.csv",
                                   "results/experiment2.csv"])
    plot_optimality_values(df, out_path="results/optimality_value.png")
"""

from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REQUIRED_BASE_COLUMNS = {"m", "n", "variance", "milp_cmax", "lpt_cmax",
                          "dp_cmax", "dp_status", "dp_is_optimal"}

# testo (sottostringa, case-insensitive) che identifica, dentro dp_note, le
# due situazioni possibili quando dp_status == "MEMORY_LIMIT_EXCEEDED" --
# vedi la docstring del modulo e quella di solvers/dp_solver.py
_NOTE_NON_ADMISSIBLE = "non e' un cmax ammissibile"
_NOTE_ADMISSIBLE_FALLBACK = "incumbent ammissibile"

# stessa palette fissa per metodo usata in plot_solver_time_by_n.py /
# plot_solver_time_by_variance.py, cosi' lo stesso metodo ha sempre lo
# stesso colore in tutta la cartella analysis/
_METHOD_COLORS = {"MILP": "tab:blue", "DP": "tab:orange", "LPT": "tab:green"}
_METHOD_ORDER = ["MILP", "DP", "LPT"]

_OOM_HATCH = "//"
_OOM_EDGE = "firebrick"
_BAR_GAP = 0.08  # spazio vuoto tra un gruppo di barre (valore di n) e il successivo, come in plot_solver_time_by_n.py

_VARIANCE_LABELS = {"low": "bassa varianza", "high": "alta varianza"}


def load_optimality_results(csv_paths: Iterable[str]) -> pd.DataFrame:
    """Concatena i CSV indicati e rimuove eventuali righe duplicate (stesso
    instance_id), stesso principio degli altri script in analysis/."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei CSV forniti: {sorted(missing)}")

    if "instance_id" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset="instance_id", keep="first")
        if len(df) < n_before:
            print(f"[plot_optimality_gap] rimosse {n_before - len(df)} righe "
                  f"duplicate (stesso instance_id in piu' file)")

    return df.sort_values(["m", "n", "variance"]).reset_index(drop=True)


def classify_dp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge una colonna 'dp_admissible' (bool): True se dp_cmax e' una
    soluzione ammissibile per l'istanza intera (ottima o sub-ottima con
    fallback), False se e' solo un cmax parziale non confrontabile (OOM
    "vero", vedi docstring del modulo). Righe con dp_status ==
    "MEMORY_LIMIT_EXCEEDED" ma dp_note non riconosciuta diventano NaN
    (escluse esplicitamente, non classificate a indovinare)."""
    df = df.copy()
    note = df.get("dp_note", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()

    is_optimal = df["dp_is_optimal"] == True  # noqa: E712 (esplicito, evita ambiguita' con NaN)
    is_admissible_fallback = note.str.contains(_NOTE_ADMISSIBLE_FALLBACK, regex=False)
    is_non_admissible = note.str.contains(_NOTE_NON_ADMISSIBLE, regex=False)

    admissible = pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    admissible[is_optimal] = True
    admissible[~is_optimal & is_admissible_fallback] = True
    admissible[~is_optimal & is_non_admissible] = False

    unrecognized = admissible.isna() & (df["dp_status"] == "MEMORY_LIMIT_EXCEEDED")
    n_unrecognized = int(unrecognized.sum())
    if n_unrecognized:
        print(f"[plot_optimality_gap] ATTENZIONE: {n_unrecognized} righe con "
              f"dp_status=MEMORY_LIMIT_EXCEEDED ma dp_note non riconosciuta "
              f"(ne' '{_NOTE_ADMISSIBLE_FALLBACK}' ne' '{_NOTE_NON_ADMISSIBLE}'): "
              f"escluse dai grafici invece di essere classificate a caso. "
              f"Controlla dp_note per queste righe.")

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
                            out_path: str = "results/optimality_value.png") -> None:
    """Un file per ciascuna coppia (m, variance) presente nei dati
    (filtrata da m_values/variance_values se forniti): un unico grafico
    con le colonne di MILP, DP e LPT affiancate per ciascun n (vedi
    docstring del modulo). Chiama internamente classify_dp_rows() se la
    colonna 'dp_admissible' non e' gia' presente (non serve chiamarla a
    mano prima). out_path e' usato come base: 'results/optimality_value.png'
    diventa 'results/optimality_value_m2_low.png', ecc."""
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
    pairs = sub[["m", "variance"]].drop_duplicates().sort_values(["m", "variance"]).itertuples(index=False)
    for m, variance in pairs:
        pair_sub = sub[(sub["m"] == m) & (sub["variance"] == variance)]
        _plot_single_pair(pair_sub, m=m, variance=variance, agg=agg, diff_labels=diff_labels,
                           out_path=f"{root}_m{m}_{variance}{ext}")


def _plot_single_pair(sub: pd.DataFrame, m: int, variance: str, agg: str, diff_labels: bool,
                       out_path: str) -> None:
    n_list = sorted(sub["n"].unique())
    if not n_list:
        print(f"[plot_optimality_gap] nessun valore di n per m={m}, variance={variance}, salto")
        return

    n_methods = len(_METHOD_ORDER)
    bar_width = (1.0 - _BAR_GAP) / n_methods
    group_positions = {n_val: i for i, n_val in enumerate(n_list)}
    # stesso schema di offset simmetrico usato in plot_solver_time_by_n.py,
    # qui applicato a MILP/DP/LPT invece che ai metodi disponibili in quel
    # modulo (qui i tre ci sono sempre, a differenza dei runtime dove un
    # metodo puo' mancare dal CSV)
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

        # --- MILP: sempre un valore normale, nessuna evidenziazione (vedi docstring) ---
        if milp_h is not None:
            ax.bar(pos_milp, milp_h, width=bar_width * 0.92, color=_METHOD_COLORS["MILP"],
                   label="_nolegend_")

        # --- DP: normale / parzialmente OOM / totalmente OOM, vedi docstring ---
        n_total = len(rows)
        n_admissible = int(rows["dp_admissible"].fillna(False).sum())
        # n_failed: conteggio GREZZO dei fallimenti (dp_status ==
        # "MEMORY_LIMIT_EXCEEDED"), usato SOLO per l'etichetta testuale.
        # Deliberatamente indipendente da dp_admissible/dp_note: una riga
        # e' un fallimento se e' andata OOM, punto -- non ci interessa qui
        # se il suo incumbent era poi classificabile come ammissibile o
        # meno (quella distinzione resta rilevante solo per decidere quali
        # righe includere nella media/mediana della barra, sotto).
        n_failed = int((rows["dp_status"] == "MEMORY_LIMIT_EXCEEDED").sum())

        if n_admissible == 0:
            # OOM totale: nessun dp_cmax confrontabile esiste per questa
            # classe. Non si inventa un valore: marcatore tratteggiato
            # all'altezza del MILP (solo per allineamento visivo, non e' un
            # dato) con etichetta esplicita "OOM (n_failed/N)".
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

        # --- LPT: sempre un valore normale, mai OOM (vedi docstring) ---
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
    ax.set_title(f"MILP vs DP vs LPT -- m={m}, {_VARIANCE_LABELS.get(variance, variance)}")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    if any_oom:
        # spazio extra in alto per non tagliare le etichette OOM/percentuali
        # appena aggiunte (l'autoscale di matplotlib considera le barre ma
        # non il testo) -- niente scala log qui (il cmax non copre ordini di
        # grandezza come i runtime), quindi il margine e' additivo e non
        # richiede attenzione all'ordine rispetto a set_yscale
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_optimality_gap] grafico salvato in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                         help="uno o piu' CSV di risultati (experiment_runner.py) da concatenare")
    parser.add_argument("--m", nargs="+", type=int, default=None,
                         help="valori di m da plottare (default: tutti quelli presenti nei CSV)")
    parser.add_argument("--variance", nargs="+", choices=["low", "high"], default=None,
                         help="valori di variance da plottare (default: entrambi)")
    parser.add_argument("--agg", choices=["median", "mean"], default="median",
                         help="aggregazione tra piu' istanze della stessa classe (m, n, variance)")
    parser.add_argument("--no-diff-labels", dest="diff_labels", action="store_false", default=True,
                         help="non mostrare l'etichetta con la differenza percentuale sopra le colonne")
    parser.add_argument("--out", default="results/optimality_value.png",
                         help="path base del file immagine di output: un file per ciascuna "
                              "combinazione (m, variance), vedi docstring del modulo")
    args = parser.parse_args()

    df = load_optimality_results(args.csv)
    df = classify_dp_rows(df)

    plot_optimality_values(df, m_values=args.m, variance_values=args.variance, agg=args.agg,
                            diff_labels=args.diff_labels, out_path=args.out)


if __name__ == "__main__":
    main()
