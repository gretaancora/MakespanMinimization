"""
Harness sperimentale: esegue i tre metodi (MILP/Gurobi, DP, euristica LPT)
su un insieme di istanze e colleziona i risultati in un DataFrame / CSV.

Uso da linea di comando:
    python experiment_runner.py --generate \
        --m 2 3 5 --n 20 50 100 --variance low high \
        --instances-per-class 10 --time-limit 120 --memory-limit-gb 2 \
        --out results/results.csv

Oppure programmaticamente:
    from experiments.experiment_runner import run_experiment
    df = run_experiment(instance_paths, time_limit=120, memory_limit_bytes=2*1024**3)
"""

from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from instances.instance_generator import Instance, generate_batch
from solvers.milp_solver import solve_milp
from solvers.dp_solver import solve_dp
from solvers.heuristic_solver import solve_lpt


def run_single(inst: Instance, time_limit: float, memory_limit_bytes: int,
               reconstruct_dp_assignment: bool = False, threads: int = 1,
               run_milp: bool = True, run_dp: bool = True, run_lpt: bool = True,
               dp_force_representation: str | None = None) -> dict:
    milp = solve_milp(inst.processing_times, inst.m, time_limit=time_limit,
                       threads=threads) if run_milp else None
    dp = solve_dp(inst.processing_times, inst.m, memory_limit_bytes=memory_limit_bytes,
                   reconstruct_assignment=reconstruct_dp_assignment,
                   force_representation=dp_force_representation) if run_dp else None
    lpt = solve_lpt(inst.processing_times, inst.m) if run_lpt else None

    # Riferimento per calcolare la qualita' relativa di DP/LPT quando il MILP
    # non ha chiuso il gap (o non e' stato eseguito): preferenza
    # MILP ottimo > DP ottimo > lower bound MILP.
    if milp is not None and milp.status == "OPTIMAL":
        reference, reference_type = milp.cmax, "MILP_OPTIMAL"
    elif dp is not None and dp.status == "OPTIMAL":
        reference, reference_type = dp.cmax, "DP_OPTIMAL"
    elif milp is not None and milp.lower_bound is not None:
        reference, reference_type = milp.lower_bound, "MILP_LOWERBOUND"
    else:
        reference, reference_type = None, "NONE"

    def rel_gap(value):
        if value is None or reference is None or reference == 0:
            return None
        return (value - reference) / reference

    # dp.cmax e' ammissibile per l'istanza intera SOLO se dp.is_optimal (vedi
    # dp_solver.py): quando la DP va in MEMORY_LIMIT_EXCEEDED, dp.cmax e' o
    # l'incumbent parziale sui soli job trattati prima del crash
    # (rappresentazione sparsa) o None (rappresentazione densa, nessun
    # incumbent) -- in NESSUNO dei due casi e' un upper bound valido per
    # l'istanza intera. Prima di questo fix, dp.cmax veniva usato cosi'
    # com'era sia in dp_gap_rel_vs_ref sia in best_known_cmax: un frammento
    # su una manciata di job e' quasi sempre numericamente piu' piccolo di
    # un cmax vero, producendo gap negativi enormi e fittizi (osservato su
    # experiment2.csv: fino a -90%) e un "miglior cmax noto" che in realta'
    # non e' una soluzione. dp_cmax resta comunque nel CSV (riga sotto,
    # invariata) per chi vuole ispezionare l'incumbent grezzo: e' solo il
    # suo USO come upper bound ammissibile a essere condizionato qui.
    dp_cmax_admissible = dp.cmax if (dp is not None and dp.is_optimal) else None

    row = {
        "instance_id": inst.id, "n": inst.n, "m": inst.m, "variance": inst.variance,
        "seed": inst.seed, "P": inst.P,
    }

    if milp is not None:
        row.update({
            "milp_status": milp.status, "milp_cmax": milp.cmax, "milp_lb": milp.lower_bound,
            "milp_gap_rel": milp.gap_rel, "milp_gap_abs": milp.gap_abs,
            "milp_runtime": milp.runtime, "milp_solver_runtime": milp.solver_runtime,
        })

    if dp is not None:
        # dp_cmax: OPTIMAL -> valore provato ottimo. MEMORY_LIMIT_EXCEEDED ->
        # incumbent ammissibile (non garantito ottimo) se disponibile, None
        # altrimenti. dp_is_optimal distingue i due casi senza dover fare
        # parsing della stringa di stato.
        row.update({
            "dp_status": dp.status, "dp_cmax": dp.cmax, "dp_is_optimal": dp.is_optimal,
            "dp_runtime": dp.runtime, "dp_solver_runtime": dp.solver_runtime,
            "dp_n_states_final": dp.n_states_final,
            "dp_n_jobs_exact": dp.n_jobs_dp_exact, "dp_note": dp.assignment_note,
            "dp_representation": dp.representation,
            "dp_gap_rel_vs_ref": rel_gap(dp_cmax_admissible),
        })

    if lpt is not None:
        row.update({
            "lpt_cmax": lpt.cmax, "lpt_runtime": lpt.runtime,
            "lpt_gap_rel_vs_ref": rel_gap(lpt.cmax),
        })

    row["reference_cmax"] = reference
    row["reference_type"] = reference_type

    # Miglior upper bound disponibile *indipendentemente dal metodo* -- utile
    # quando la DP va in MEMORY_LIMIT_EXCEEDED: il suo incumbent non e' detto
    # sia migliore di LPT (il completamento greedy sui job non trattati dalla
    # DP puo' essere peggiore di un LPT globale), va quindi preso il minimo
    # tra i due disponibili, non dato per scontato che la DP vinca sempre.
    # Usa dp_cmax_admissible, NON dp.cmax grezzo: altrimenti un incumbent
    # parziale non ammissibile (vedi sopra) finirebbe scelto come "migliore"
    # solo perche' numericamente piccolo, non perche' sia una soluzione.
    candidates = [v for v in [dp_cmax_admissible,
                               lpt.cmax if lpt is not None else None] if v is not None]
    if candidates:
        row["best_known_cmax"] = min(candidates)

    return row


def run_experiment(instance_paths, time_limit: float = 120.0,
                    memory_limit_bytes: int = 16 * 1024**3,
                    out_csv: str = None, verbose: bool = True, threads: int = 1,
                    run_milp: bool = True, run_dp: bool = True, run_lpt: bool = True,
                    dp_force_representation: str | None = None) -> pd.DataFrame:
    if run_dp:
        # Warm-up: la primissima chiamata a solve_dp in un interprete nuovo puo'
        # costare sensibilmente di piu' (anche +50-100%) della norma -- probabilmente
        # per costi di inizializzazione a livello di sistema/librerie (BLAS, cache
        # delle pagine, allocatore) che vengono "pagati" una volta sola e non sono
        # rappresentativi dell'algoritmo. L'effetto non e' deterministico ne'
        # quantificabile come una singola costante da sottrarre (misurato: a volte
        # non si presenta affatto), quindi non proviamo a stimarlo: ci limitiamo a
        # "scaricarlo" qui su un'istanza fittizia, scartata, cosi' non inquina la
        # prima riga vera del CSV. Copre sia il path denso che quello sparso.
        dummy_p = [3, 5, 2, 4]
        solve_dp(dummy_p, 2, memory_limit_bytes=memory_limit_bytes,
                 reconstruct_assignment=False, force_representation="dense")
        solve_dp(dummy_p, 4, memory_limit_bytes=memory_limit_bytes,
                 reconstruct_assignment=False, force_representation="sparse")

    rows = []
    total = len(instance_paths)
    for k, path in enumerate(instance_paths, 1):
        inst = Instance.load(path)
        t0 = time.time()
        row = run_single(inst, time_limit, memory_limit_bytes, threads=threads,
                          run_milp=run_milp, run_dp=run_dp, run_lpt=run_lpt,
                          dp_force_representation=dp_force_representation)
        rows.append(row)
        if verbose:
            parts = [f"[{k}/{total}] {inst.id}"]
            if "milp_status" in row:
                parts.append(f"MILP={row['milp_status']}({row['milp_runtime']:.1f}s)")
            if "dp_status" in row:
                parts.append(f"DP={row['dp_status']}({row['dp_runtime']:.1f}s)")
            if "lpt_gap_rel_vs_ref" in row:
                parts.append(f"LPT_gap={row['lpt_gap_rel_vs_ref']}")
            parts.append(f"[wall {time.time()-t0:.1f}s]")
            print("  ".join(parts))
        if out_csv:
            pd.DataFrame(rows).to_csv(out_csv, index=False)  # salvataggio incrementale
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                     help="genera nuove istanze prima di eseguire la campagna")
    ap.add_argument("--m", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--n", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--variance", type=str, nargs="+", default=["low", "high"],
                     choices=["low", "high"])
    ap.add_argument("--instances-per-class", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--memory-limit-gb", type=float, default=16.0)
    ap.add_argument("--threads", type=int, default=1,
                     help="thread Gurobi (default 1: risultati riproducibili e "
                          "confrontabili con la DP, che gira su un solo core)")
    ap.add_argument("--instances-dir", type=str, default="instances/data")
    ap.add_argument("--out", type=str, default="results/results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-milp", action="store_true", help="non eseguire il MILP")
    ap.add_argument("--skip-dp", action="store_true", help="non eseguire la DP")
    ap.add_argument("--skip-lpt", action="store_true", help="non eseguire l'euristica LPT")
    ap.add_argument("--dp-representation", type=str, default=None,
                     choices=["dense", "sparse"],
                     help="forza la rappresentazione usata dalla DP (dense/sparse) "
                          "bypassando la scelta automatica; default: automatico")
    args = ap.parse_args()

    if args.generate:
        paths = generate_batch(args.n, args.m, args.variance,
                                args.instances_per_class, args.instances_dir,
                                base_seed=args.seed)
    else:
        paths = [os.path.join(args.instances_dir, f)
                  for f in sorted(os.listdir(args.instances_dir)) if f.endswith(".json")]

    print(f"Eseguo la campagna su {len(paths)} istanze "
          f"(time_limit={args.time_limit}s, memory_limit={args.memory_limit_gb}GB, "
          f"threads={args.threads}, "
          f"metodi={'MILP ' if not args.skip_milp else ''}"
          f"{'DP ' if not args.skip_dp else ''}{'LPT' if not args.skip_lpt else ''}, "
          f"dp_representation={args.dp_representation or 'auto'})")
    df = run_experiment(paths, time_limit=args.time_limit,
                         memory_limit_bytes=int(args.memory_limit_gb * 1024**3),
                         out_csv=args.out, threads=args.threads,
                         run_milp=not args.skip_milp, run_dp=not args.skip_dp,
                         run_lpt=not args.skip_lpt,
                         dp_force_representation=args.dp_representation)
    print(f"Risultati salvati in {args.out} ({len(df)} righe)")


if __name__ == "__main__":
    main()
