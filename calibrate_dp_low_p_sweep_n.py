"""
Sweep fine di n tra l'ultimo n per cui la DP chiude OPTIMAL (almeno una
istanza) e il primo n per cui fallisce sistematicamente (nessuna
istanza), per ciascun m -- rifinisce la stima grezza prodotta dalla
crescita moltiplicativa di calibrate_dp_low_p_scaling_n.py. Quello script
procede a salti (--growth-factor, default raddoppio): il confine vero
puo' cadere ovunque dentro l'ultimo salto, non necessariamente vicino a
uno dei due estremi -- ad es. nel CSV di esempio, per m=3 il salto va da
n=2560 (3/3 OPTIMAL) a n=5120 (0/3 OPTIMAL): il confine reale e' ignoto
ovunque in quell'intervallo di 2560 valori.

Input: il CSV di DETTAGLIO di calibrate_dp_low_p_scaling_n.py (non il
summary -- servono le singole righe per ricalcolare il bracket [ultimo n
con almeno un OPTIMAL, primo n con zero OPTIMAL] per ciascun m, con
ESATTAMENTE la stessa logica di _build_summary, riusata qui via import
diretto invece di duplicarla -- cosi' il bracket coincide sempre con
quello che calibrate_dp_low_p_scaling_n.py avrebbe riportato nel proprio
CSV di riepilogo).

--representation deve corrispondere a come e' stato generato il CSV
fornito (default "auto", lo stesso default dello script che lo produce):
se il CSV viene da una run con --representation both, il bracket viene
ricalcolato separatamente per sparse e dense (stessa logica del
riepilogo in modalita' both), e ciascuna meta' dello sweep forza la
rispettiva rappresentazione; altrimenti (auto/sparse/dense) c'e' un solo
bracket per m e si usa la stessa force_representation della run
originale (None per "auto").

Due modalita' (--mode):
  - "linear" (default): --n-points valori di n dentro il bracket
    (esclusi gli estremi, che sono gia' noti), equispaziati -- linear-
    mente, oppure logaritmicamente con --log-spacing (ha senso quando il
    bracket e' ampio, dato che a monte la crescita e' moltiplicativa: gli
    spazi lineari sprecherebbero risoluzione nella coda alta). Utile per
    vedere la FORMA della transizione, che non e' detto sia un gradino
    netto: nel CSV di esempio, m=5 mostra gia' 1/3 istanze ancora OPTIMAL
    a n=80 pur avendo 3/3 fallite a n=160 -- una transizione graduale, non
    un singolo punto di rottura.
  - "bisect": bisezione classica sul bracket (dimezza ogni volta in base
    a se ALMENO UNA istanza chiude OPTIMAL al punto medio), finche' non
    resta piu' stretto di --resolution -- individua il confine con molte
    meno valutazioni (O(log2(ampiezza)) invece di --n-points fisso), ma
    assume implicitamente una transizione a gradino netto: se e' invece
    graduale (come sopra), la bisezione restituisce comunque un punto,
    ma quel punto non riassume la gradualita' -- usare "linear" se questo
    conta.

--memory-limit-gb, --time-soft-cap-s, --instances-per-class e
--p-low/--p-high NON sono ricavabili dal CSV di dettaglio (non ce li
riporta): vanno ripassati identici a quelli usati per generare il CSV di
input, altrimenti lo sweep non e' confrontabile con i punti gia' noti
(--p-low/--p-high vengono pero' letti direttamente dal CSV per sicurezza,
dato che sono comunque presenti come colonne, cosi' non serve
ridigitarli). --base-seed ha un default diverso dagli altri script di
calibrazione per non generare accidentalmente le stesse istanze gia'
misurate ai due estremi del bracket.

Uso:
    python3 calibrate_dp_low_p_sweep_n.py --csv results/calibration_dp_low_p_scaling_n.csv
        [--representation auto] [--mode linear] [--n-points 8] [--log-spacing]
        [--instances-per-class 3] [--memory-limit-gb 16] [--time-soft-cap-s 120]

Oppure per la bisezione:
    python3 calibrate_dp_low_p_sweep_n.py --csv ... --mode bisect --resolution 1
"""
import argparse
import math
import sys, os, csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from instances.instance_generator import generate_instance_custom_range
from solvers.dp_solver import solve_dp, choose_representation
from solvers.heuristic_solver import solve_lpt
from calibrate_dp_low_p_scaling_n import _build_summary

REQUIRED_COLUMNS = {"m", "n", "dp_status", "stop_reason", "representation", "p_low", "p_high"}


def _load_detail_rows(csv_path: str):
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti in {csv_path}: {sorted(missing)} "
                          f"(serve il CSV di DETTAGLIO di calibrate_dp_low_p_scaling_n.py, "
                          f"non quello di riepilogo)")
    records = df.to_dict("records")
    # df.where(pd.notnull(df), None) NON basta: su colonne miste pandas spesso
    # reintroduce NaN (float) invece di None durante il cast implicito. _build_summary
    # si aspetta "is None", non un controllo isna(), quindi sostituiamo a mano.
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


def _brackets_per_group(rows, representation_mode, allow_time_cap_only=False):
    """Ricalcola, con la stessa logica di _build_summary, il bracket [ultimo n
    con >=1 OPTIMAL, primo n con 0 OPTIMAL] per ciascun m (o ciascun (m,
    representation) se representation_mode == 'both'). Ritorna una lista di
    tuple (m, force_representation_o_None, low, high); salta silenziosamente
    (con un avviso) i gruppi il cui bracket e' incompleto (mai fallita entro
    l'n massimo testato), gia' stretto (nessun affinamento possibile), o --
    salvo allow_time_cap_only=True -- il cui fallimento non e' MAI dovuto a
    MEMORY_LIMIT_EXCEEDED in nessuna riga del gruppo (solo
    TIME_SOFT_CAP_STOPPED): in quel caso il "confine" che si affinerebbe non
    e' un limite di memoria ma solo la soglia --time-soft-cap-s scelta a
    monte, un artefatto della calibrazione e non una proprieta' della DP --
    affinarlo non direbbe nulla sul comportamento esponenziale in memoria che
    e' l'oggetto di questa campagna (vedi docstring di
    calibrate_dp_low_p_many_m.py)."""
    summary = _build_summary(rows, representation_mode)
    by_group = defaultdict(list)
    for r in rows:
        key = (r["m"], r["representation"]) if representation_mode == "both" else (r["m"],)
        by_group[key].append(r)

    brackets = []
    for s in summary:
        label = f"m={s['m']}" + (f" representation={s['representation']}" if representation_mode == "both" else "")
        if s["last_n_some_optimal"] is None or s["first_n_all_failed"] is None:
            print(f"[sweep] {label}: bracket incompleto ({s['recap']}), salto")
            continue
        low, high = s["last_n_some_optimal"], s["first_n_all_failed"]
        if high - low <= 1:
            print(f"[sweep] {label}: bracket gia' stretto ({low}-{high}), niente da affinare, salto")
            continue

        if not allow_time_cap_only:
            key = (s["m"], s["representation"]) if representation_mode == "both" else (s["m"],)
            group_rows = by_group[key]
            has_real_oom = any(r["dp_status"] == "MEMORY_LIMIT_EXCEEDED" for r in group_rows)
            if not has_real_oom:
                print(f"[sweep] {label}: nessun MEMORY_LIMIT_EXCEEDED nel gruppo, il fallimento a "
                      f"n={high} e' solo TIME_SOFT_CAP_STOPPED (non e' un limite di memoria, e' il "
                      f"soft cap di tempo scelto a monte), salto -- passare --allow-time-cap-only "
                      f"per affinarlo comunque")
                continue

        force = None if representation_mode == "auto" else (
            s["representation"] if representation_mode == "both" else representation_mode)
        brackets.append((s["m"], force, low, high))
    return brackets


def _linear_points(low: int, high: int, n_points: int, log_spacing: bool) -> list:
    if n_points < 1:
        return []
    if log_spacing:
        log_low, log_high = math.log(low), math.log(high)
        raw = [math.exp(log_low + (log_high - log_low) * i / (n_points + 1)) for i in range(1, n_points + 1)]
    else:
        raw = [low + (high - low) * i / (n_points + 1) for i in range(1, n_points + 1)]
    # arrotonda e vincola strettamente dentro (low, high): gli estremi sono gia' noti,
    # non ha senso rimisurarli; dedup preserva l'ordine
    seen = []
    for x in raw:
        v = min(high - 1, max(low + 1, round(x)))
        if v not in seen:
            seen.append(v)
    return seen


def _run_one(m: int, n: int, seed: int, p_low: int, p_high: int, force_representation,
             memory_limit_bytes: int, time_soft_cap: float):
    """Esegue una singola istanza (m, n, seed) e ritorna (row_dict, is_optimal).
    is_optimal e' True solo se dp_status == OPTIMAL E non ha sforato il soft
    cap di tempo, stessa convenzione di calibrate_dp_low_p_scaling_n.py."""
    inst = generate_instance_custom_range(n=n, m=m, seed=seed, p_low=p_low, p_high=p_high)
    lpt = solve_lpt(inst.processing_times, m)
    auto_pick = choose_representation(inst.P, m - 1, memory_limit_bytes)
    dp = solve_dp(inst.processing_times, m, memory_limit_bytes=memory_limit_bytes,
                   reconstruct_assignment=False, force_representation=force_representation)

    stop_reason = None
    if dp.status == "OPTIMAL" and dp.runtime > time_soft_cap:
        stop_reason = "TIME_SOFT_CAP_STOPPED"
    elif dp.status == "MEMORY_LIMIT_EXCEEDED":
        stop_reason = "MEMORY_LIMIT_EXCEEDED"
    elif dp.status == "ERROR":
        stop_reason = "ERROR"
    is_optimal = dp.status == "OPTIMAL" and stop_reason is None

    rep_used = dp.representation or force_representation or "auto"
    mem_str = f"{dp.peak_memory_mb:.1f}" if dp.peak_memory_mb is not None else "n/a"
    vsz_str = f"{dp.peak_vsz_mb:.1f}" if dp.peak_vsz_mb is not None else "n/a"
    dp_cmax_str = str(dp.cmax) if dp.cmax is not None else "n/a"
    print(f"{m:3d} {n:9d} {seed:7d} {rep_used:>6s} {auto_pick:>6s}  {dp.status:>22s} "
          f"{dp.runtime:10.3f} {mem_str:>10s} {vsz_str:>10s} {dp.n_states_final:12d} "
          f"{lpt.cmax:8d} {dp_cmax_str:>8s}" + (f"   <- {stop_reason}" if stop_reason else ""))

    row = dict(m=m, n=n, seed=seed, p_low=p_low, p_high=p_high, representation=rep_used,
               auto_pick=auto_pick, dp_status=dp.status, dp_cmax=dp.cmax, reference_cmax=lpt.cmax,
               dp_runtime=round(dp.runtime, 3),
               dp_peak_memory_mb=round(dp.peak_memory_mb, 1) if dp.peak_memory_mb is not None else None,
               dp_peak_vsz_mb=round(dp.peak_vsz_mb, 1) if dp.peak_vsz_mb is not None else None,
               dp_states=dp.n_states_final, lpt_runtime=round(lpt.runtime, 5), stop_reason=stop_reason)
    return row, is_optimal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, required=True,
                     help="CSV di DETTAGLIO prodotto da calibrate_dp_low_p_scaling_n.py")
    ap.add_argument("--representation", choices=["auto", "sparse", "dense", "both"], default="auto",
                     help="deve corrispondere a come e' stato generato --csv, vedi docstring")
    ap.add_argument("--allow-time-cap-only", action="store_true",
                     help="di default un bracket viene saltato se nessuna riga del gruppo ha mai "
                          "MEMORY_LIMIT_EXCEEDED (fallimento solo per TIME_SOFT_CAP_STOPPED, non un "
                          "limite di memoria); passare questo flag per affinarlo comunque")
    ap.add_argument("--mode", choices=["linear", "bisect"], default="linear")
    ap.add_argument("--n-points", type=int, default=8,
                     help="[--mode linear] quanti n intermedi testare dentro ciascun bracket")
    ap.add_argument("--log-spacing", action="store_true",
                     help="[--mode linear] spaziatura logaritmica invece che lineare, vedi docstring")
    ap.add_argument("--resolution", type=int, default=1,
                     help="[--mode bisect] la bisezione si ferma quando il bracket e' largo al piu' "
                          "cosi' (default 1: individua l'esatto n di confine)")
    ap.add_argument("--instances-per-class", type=int, default=3,
                     help="deve corrispondere a quello usato per generare --csv")
    ap.add_argument("--memory-limit-gb", type=float, default=16.0,
                     help="deve corrispondere a quello usato per generare --csv")
    ap.add_argument("--time-soft-cap-s", type=float, default=120.0,
                     help="deve corrispondere a quello usato per generare --csv")
    ap.add_argument("--out", type=str, default="results/calibration_dp_low_p_sweep_n.csv")
    ap.add_argument("--base-seed", type=int, default=13000,
                     help="seed di partenza, distinto dagli altri script di calibrazione per non "
                          "rigenerare per caso le stesse istanze gia' misurate ai due estremi")
    args = ap.parse_args()

    memory_limit_bytes = int(args.memory_limit_gb * 1024**3)
    detail_rows = _load_detail_rows(args.csv)
    p_low = detail_rows[0]["p_low"]
    p_high = detail_rows[0]["p_high"]

    brackets = _brackets_per_group(detail_rows, args.representation, args.allow_time_cap_only)
    if not brackets:
        print("\nnessun bracket da affinare (vedi avvisi sopra), esco")
        return

    print(f"\np in [{p_low},{p_high}]  representation={args.representation}  mode={args.mode}  "
          f"memory_limit={args.memory_limit_gb}GB  time_soft_cap={args.time_soft_cap_s}s\n")
    print(f"{'m':>3s} {'n':>9s} {'seed':>7s} {'rep':>6s} {'auto':>6s}  {'dp_status':>22s} "
          f"{'dp_runtime':>10s} {'dp_rss_mb':>10s} {'dp_vsz_mb':>10s} {'dp_states':>12s} "
          f"{'lpt_cmax':>8s} {'dp_cmax':>8s}")

    rows = []
    bisect_results = []  # solo per --mode bisect, per il riepilogo finale

    for m, force_rep, low, high in brackets:
        label = f"m={m}" + (f" rep={force_rep}" if args.representation == "both" else "")
        print(f"\n-- {label}: bracket iniziale [{low}, {high}] --")

        if args.mode == "linear":
            ns = _linear_points(low, high, args.n_points, args.log_spacing)
            for n in ns:
                statuses_this_n = []
                for k in range(args.instances_per_class):
                    seed = args.base_seed + 1_000_000 * m + 1_000 * n + k
                    row, _ = _run_one(m, n, seed, p_low, p_high, force_rep,
                                       memory_limit_bytes, args.time_soft_cap_s)
                    rows.append(row)
                    statuses_this_n.append(row["dp_status"])
                if statuses_this_n and all(s == "MEMORY_LIMIT_EXCEEDED" for s in statuses_this_n):
                    print(f"   {label}: {args.instances_per_class}/{args.instances_per_class} istanze "
                          f"in MEMORY_LIMIT_EXCEEDED a n={n}, mi fermo qui (non valuto gli n piu' "
                          f"grandi ancora previsti per questo m: sono comunque oltre il punto in cui "
                          f"e' gia' out-of-memory su tutta la classe, non aggiungono informazione e "
                          f"costano solo tempo)")
                    break

        else:  # bisect
            cur_low, cur_high = low, high
            k = 0
            while cur_high - cur_low > args.resolution:
                mid = (cur_low + cur_high) // 2
                if mid == cur_low:
                    break  # non si puo' piu' stringere con divisione intera
                n_optimal = 0
                for j in range(args.instances_per_class):
                    seed = args.base_seed + 1_000_000 * m + 1_000 * mid + j + 1000 * k
                    row, is_optimal = _run_one(m, mid, seed, p_low, p_high, force_rep,
                                                memory_limit_bytes, args.time_soft_cap_s)
                    rows.append(row)
                    if is_optimal:
                        n_optimal += 1
                if n_optimal > 0:
                    cur_low = mid
                else:
                    cur_high = mid
                k += 1
            print(f"   bracket affinato: [{cur_low}, {cur_high}]")
            bisect_results.append(dict(m=m, representation=force_rep or "auto",
                                        bracket_iniziale=f"{low}-{high}",
                                        bracket_finale=f"{cur_low}-{cur_high}"))

    if not rows:
        print("\nnessuna riga prodotta")
        return

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSweep salvato in {out_path} ({len(rows)} righe)")
    print("(stesso schema di colonne del CSV di calibrate_dp_low_p_scaling_n.py: puoi concatenarli "
          "per ottenere una vista unica piu' densa nella zona di transizione)")

    if bisect_results:
        print("\nbracket affinati per bisezione:")
        for r in bisect_results:
            print(f"  m={r['m']:<3d} representation={r['representation']:<8s} "
                  f"{r['bracket_iniziale']:>12s} -> {r['bracket_finale']}")


if __name__ == "__main__":
    main()
