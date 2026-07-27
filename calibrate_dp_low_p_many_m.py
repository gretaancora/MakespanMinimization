"""
Calibrazione DP: tempi di processamento "pochi unita'" e molte macchine,
a parita' di ordine di grandezza del Cmax atteso.

Motivazione
-----------
Le calibrazioni in calibrate_dp_lpt.py fanno variare n a m fisso, per
capire dove la DP comincia a soffrire man mano che l'istanza cresce. Qui
la domanda e' diversa e complementare: la DP (rappresentazione sparsa,
vedi dp_solver.py) e' polinomiale in P = sum(p_i) ma ESPONENZIALE nel
numero di dimensioni dims = m-1, perche' lo spazio degli stati teorico e'
(P+1)^dims. Nelle calibrazioni gia' fatte i due effetti (P grande, m
grande) sono spesso confusi insieme (es. "alta varianza" = job fino a
1000, che gia' da solo gonfia P). Qui si isola l'effetto di m tenendo P
*basso* per costruzione:

  - tempi di processamento uniformi in [--p-low, --p-high], di default
    [1, 5] ("poche unita'");
  - per ciascun valore di m nella griglia, si sceglie n in modo che il
    Cmax atteso (~ P/m per job piccoli rispetto al carico per macchina)
    resti in una fascia bassa e fissa (default centrata su --target-cmax,
    pensata per restare nell'ordine "2-3 decine" cioe' 20-30): n ~=
    target_cmax * m / avg_p. Di conseguenza, al crescere di m cresce
    anche n (e quindi P) in proporzione, ma il VALORE della soluzione
    resta sempre nello stesso piccolo intorno -- solo dims cresce.

Il punto e' proprio verificare se, a parita' di "difficolta' apparente"
del problema (soluzione ottima piccola, tempi piccoli), la DP smette
comunque di chiudere all'ottimo (tempo e/o memoria) semplicemente perche'
m cresce: se succede, e' evidenza diretta della componente esponenziale
in m, non spiegabile con P grande (qui P resta piccolo per costruzione).

Uso:
    python3 calibrate_dp_low_p_many_m.py [--p-low 1] [--p-high 5]
        [--target-cmax 25] [--m-values 3 4 5 6]
        [--instances-per-class 3] [--memory-limit-gb 16]
        [--time-soft-cap-s 120]

Ogni riga del CSV di dettaglio (--out) e' una singola istanza (m, seed):
n scelto, Cmax atteso (target) vs Cmax osservato (LPT, sempre disponibile
come riferimento ammissibile) vs esito DP (dp_status, dp_cmax quando
disponibile, dp_runtime, dp_states, dp_peak_memory_mb, dp_peak_vsz_mb). Il
CSV di riepilogo (--summary-out) aggrega per m: quante istanze su
instances-per-class la DP ha chiuso OPTIMAL, e a partire da che m smette
sistematicamente di chiudere.

Nota sulla rappresentazione: qui si usa la scelta AUTOMATICA di solve_dp
(force_representation=None), non forzata, perche' l'obiettivo e' osservare
il comportamento "reale" che si avrebbe in campagna, non confrontare le
due rappresentazioni tra loro (per quello vedi calibrate_dp_lpt.py). In
pratica, dato che _use_dense in dp_solver.py rifiuta comunque dims>5, per
m>=7 la scelta automatica ricade sempre sulla sparsa.
"""
import argparse
import sys, os, csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from instances.instance_generator import generate_instance_custom_range
from solvers.dp_solver import solve_dp, choose_representation
from solvers.heuristic_solver import solve_lpt

DEFAULT_M_VALUES = [3, 4, 5, 6]


def _target_n(m: int, target_cmax: float, avg_p: float) -> int:
    # n job ~= target_cmax * m / avg_p, con almeno m+1 job (altrimenti con
    # n<=m ogni job puo' finire su una macchina diversa e il Cmax atteso
    # non e' piu' rappresentativo di P/m ma del singolo job piu' lungo)
    n = round(target_cmax * m / avg_p)
    return max(n, m + 1)


def _build_summary(rows):
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["m"]].append(r)

    summary = []
    first_m_not_all_optimal = None
    for m in sorted(by_m.keys()):
        rs = by_m[m]
        n_optimal = sum(1 for r in rs if r["dp_status"] == "OPTIMAL")
        n_total = len(rs)
        avg_cmax_target = sum(r["target_cmax"] for r in rs) / n_total
        avg_cmax_observed = sum(r["reference_cmax"] for r in rs) / n_total
        avg_dp_runtime = (sum(r["dp_runtime"] for r in rs if r["dp_status"] == "OPTIMAL") / n_optimal
                           if n_optimal else None)
        if n_optimal < n_total and first_m_not_all_optimal is None:
            first_m_not_all_optimal = m
        summary.append(dict(
            m=m, dims=m - 1, n_instances=n_total,
            n_optimal=n_optimal,
            frac_optimal=round(n_optimal / n_total, 2),
            avg_target_cmax=round(avg_cmax_target, 1),
            avg_reference_cmax=round(avg_cmax_observed, 1),
            avg_dp_runtime_s=(round(avg_dp_runtime, 3) if avg_dp_runtime is not None else None),
            n_sample=rs[0]["n"],
        ))
    recap = ("la DP chiude OPTIMAL su tutta la griglia testata (nessun m ha fatto fallire nessuna istanza)"
              if first_m_not_all_optimal is None else
              f"dal primo m={first_m_not_all_optimal} in poi la DP smette di chiudere OPTIMAL su tutte "
              f"le istanze, pur restando il Cmax atteso nell'ordine di {round(sum(r['target_cmax'] for r in rows) / len(rows))} "
              f"(evidenza della componente esponenziale in m = dims+1, non spiegabile da P grande)")
    return summary, recap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-low", type=int, default=1, help="estremo inferiore range tempi (incluso)")
    ap.add_argument("--p-high", type=int, default=5, help="estremo superiore range tempi (incluso)")
    ap.add_argument("--target-cmax", type=float, default=25.0,
                     help="Cmax atteso da centrare (default 25, pensato per restare in 20-30)")
    ap.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
                     help="griglia di numero di macchine da testare")
    ap.add_argument("--instances-per-class", type=int, default=3,
                     help="quante istanze (seed diversi) per ciascun valore di m")
    ap.add_argument("--memory-limit-gb", type=float, default=16.0,
                     help="deve corrispondere al budget usato nella campagna finale")
    ap.add_argument("--time-soft-cap-s", type=float, default=120.0,
                     help="oltre questo tempo per una singola istanza, si segnala "
                          "TIME_SOFT_CAP_STOPPED (non un fallimento della DP, vedi calibrate_dp_lpt.py "
                          "per la stessa convenzione) e si smette di salire in m dopo due valori "
                          "consecutivi in cui nessuna istanza chiude OPTIMAL entro il soft cap")
    ap.add_argument("--out", type=str, default="results/calibration_dp_low_p_many_m.csv")
    ap.add_argument("--summary-out", type=str,
                     default="results/calibration_dp_low_p_many_m_summary.csv")
    ap.add_argument("--base-seed", type=int, default=5000,
                     help="seed di partenza (offset per non collidere con gli altri script di calibrazione)")
    args = ap.parse_args()

    if args.p_low < 1 or args.p_high < args.p_low:
        ap.error(f"range non valido: --p-low {args.p_low} --p-high {args.p_high}")

    memory_limit_bytes = int(args.memory_limit_gb * 1024**3)
    avg_p = (args.p_low + args.p_high) / 2.0

    rows = []
    print(f"p in [{args.p_low},{args.p_high}] (avg={avg_p:.1f})  target_cmax={args.target_cmax}  "
          f"memory_limit={args.memory_limit_gb}GB  time_soft_cap={args.time_soft_cap_s}s\n")
    print(f"{'m':>3s} {'dims':>4s} {'n':>4s} {'seed':>6s} {'auto':>6s}  {'dp_status':>22s} "
          f"{'dp_runtime':>10s} {'dp_rss_mb':>10s} {'dp_vsz_mb':>10s} {'dp_states':>12s} "
          f"{'lpt_cmax':>8s} {'dp_cmax':>8s}")

    n_m_fully_stuck_in_a_row = 0
    for m in args.m_values:
        n = _target_n(m, args.target_cmax, avg_p)
        n_optimal_this_m = 0
        for k in range(args.instances_per_class):
            seed = args.base_seed + 1000 * m + k
            inst = generate_instance_custom_range(n=n, m=m, seed=seed,
                                                    p_low=args.p_low, p_high=args.p_high)
            lpt = solve_lpt(inst.processing_times, m)
            auto_pick = choose_representation(inst.P, m - 1, memory_limit_bytes)

            dp = solve_dp(inst.processing_times, m, memory_limit_bytes=memory_limit_bytes,
                           reconstruct_assignment=False, force_representation=None)

            stop_reason = None
            if dp.status == "OPTIMAL" and dp.runtime > args.time_soft_cap_s:
                stop_reason = "TIME_SOFT_CAP_STOPPED"
            elif dp.status == "MEMORY_LIMIT_EXCEEDED":
                stop_reason = "MEMORY_LIMIT_EXCEEDED"
            elif dp.status == "ERROR":
                stop_reason = "ERROR"
            if dp.status == "OPTIMAL" and stop_reason is None:
                n_optimal_this_m += 1

            mem_str = f"{dp.peak_memory_mb:.1f}" if dp.peak_memory_mb is not None else "n/a"
            vsz_str = f"{dp.peak_vsz_mb:.1f}" if dp.peak_vsz_mb is not None else "n/a"
            dp_cmax_str = str(dp.cmax) if dp.cmax is not None else "n/a"
            print(f"{m:3d} {m - 1:4d} {n:4d} {seed:6d} {auto_pick:>6s}  {dp.status:>22s} "
                  f"{dp.runtime:10.3f} {mem_str:>10s} {vsz_str:>10s} {dp.n_states_final:12d} "
                  f"{lpt.cmax:8d} {dp_cmax_str:>8s}"
                  + (f"   <- {stop_reason}" if stop_reason else ""))

            rows.append(dict(m=m, n=n, seed=seed, p_low=args.p_low, p_high=args.p_high,
                              target_cmax=args.target_cmax, reference_cmax=lpt.cmax,
                              auto_pick=auto_pick,
                              dp_status=dp.status, dp_cmax=dp.cmax,
                              dp_runtime=round(dp.runtime, 3),
                              dp_peak_memory_mb=(round(dp.peak_memory_mb, 1)
                                                  if dp.peak_memory_mb is not None else None),
                              dp_peak_vsz_mb=(round(dp.peak_vsz_mb, 1)
                                              if dp.peak_vsz_mb is not None else None),
                              dp_states=dp.n_states_final,
                              lpt_cmax=lpt.cmax, lpt_runtime=round(lpt.runtime, 5),
                              stop_reason=stop_reason))

        if n_optimal_this_m == 0:
            n_m_fully_stuck_in_a_row += 1
        else:
            n_m_fully_stuck_in_a_row = 0
        if n_m_fully_stuck_in_a_row >= 2:
            print(f"\n(mi fermo: nessuna istanza ha chiuso OPTIMAL entro il soft cap per due valori "
                  f"di m consecutivi, l'ultimo essendo m={m})")
            break

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDettaglio salvato in {out_path} ({len(rows)} righe)")

    summary, recap = _build_summary(rows)
    summary_path = os.path.join(os.path.dirname(__file__), args.summary_out)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Riepilogo salvato in {summary_path}\n")

    print(f"{'m':>3s} {'dims':>4s} {'n':>4s} {'ottime/tot':>10s} {'target Cmax':>12s} "
          f"{'Cmax osserv.':>13s} {'dp_runtime medio':>17s}")
    for s in summary:
        print(f"{s['m']:3d} {s['dims']:4d} {s['n_sample']:4d} "
              f"{str(s['n_optimal']) + '/' + str(s['n_instances']):>10s} "
              f"{s['avg_target_cmax']:12.1f} {s['avg_reference_cmax']:13.1f} "
              f"{str(s['avg_dp_runtime_s']):>17s}")
    print(f"\n{recap}")


if __name__ == "__main__":
    main()
