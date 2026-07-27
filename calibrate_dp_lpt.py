"""
Calibrazione DP + LPT (MILP escluso qui: va calibrato separatamente con
Gurobi disponibile, vedi README). Ha due scopi, entrambi utili prima della
campagna sperimentale vera e propria:

1. Capire a partire da quale n la DP comincia a soffrire (tempo e/o
   memoria) per ciascuna combinazione (m, variance), cosi' da poter
   scegliere valori di n "piccolo / medio / grande" sensati anche dal lato
   DP, non solo dal lato MILP.

2. Confrontare le DUE RAPPRESENTAZIONI della DP (densa/numpy e
   sparsa/set-di-tuple, vedi dp_solver.py) *sulla stessa istanza*, per
   capire empiricamente quando conviene forzare l'una o l'altra invece di
   affidarsi alla stima automatica (`choose_representation`/`_use_dense`
   in dp_solver.py). Per ogni (m, variance, n) lo script forza entrambe le
   rappresentazioni con `solve_dp(..., force_representation=...)` e
   registra tempo, stati ed esito per ciascuna, in modo indipendente:
   se una rappresentazione va in memory limit o supera il soft cap di
   tempo, si smette di scalare in n per QUELLA rappresentazione ma si
   continua con l'altra finche' anche lei non si ferma. Alla fine viene
   prodotto anche un riepilogo per classe (m, variance) con lo speedup
   osservato densa/sparsa e una raccomandazione pratica su fino a che n
   conviene forzare la densa.

Uso: python3 calibrate_dp_lpt.py [--memory-limit-gb 16] [--time-soft-cap-s 120]
                                  [--representation {sparse,dense,both}]

Per default (--representation both, invariato rispetto a prima) lo script
forza ed esegue ENTRAMBE le rappresentazioni per ogni istanza, come
descritto sopra. Passando --representation sparse oppure --representation
dense si forza e si esegue SOLO quella, dimezzando il numero di solve (e
quindi il tempo di calibrazione): utile quando si vuole solo capire dove
si ferma una rappresentazione specifica, senza pagare il costo di
eseguire anche l'altra. In questo caso il confronto densa-vs-sparsa (e lo
speedup) non ha senso -- il riepilogo lo riflette: niente colonne di
speedup, solo fino a che n quella rappresentazione ha chiuso OPTIMAL e
dove si e' fermata (vedi _build_summary).

IMPORTANTE: il limite di memoria qui va allineato a quello che userete
davvero nella campagna finale (--memory-limit-gb di experiment_runner.py,
default 16). Passarlo esplicitamente invece di lasciare un valore
hardcoded diverso nello script evita il problema riscontrato in una
versione precedente di questo file (costante fissa a 1.5GB, mai allineata
al default del progetto -> risultati di calibrazione non rappresentativi
del budget di memoria realmente usato in campagna).

dp_status nel CSV distingue tre casi, non due:
  - OPTIMAL: la DP (in quella rappresentazione) ha chiuso entro il budget
    di memoria dato
  - MEMORY_LIMIT_EXCEEDED: la DP ha esaurito davvero la memoria in quella
    rappresentazione (vedi dp_solver.py per l'incumbent che restituirebbe
    comunque in campagna vera; qui non ricostruito -- si passa
    reconstruct_assignment=False per la sola calibrazione del valore
    ottimo). Per la rappresentazione densa questo puo' succedere anche
    quando la stima automatica *non* l'avrebbe scelta (qui la si forza
    comunque, apposta, per scoprire il limite pratico reale invece di
    quello solo stimato -- vedi note su force_representation in
    dp_solver.py).
  - TIME_SOFT_CAP_STOPPED: NON e' un fallimento della DP -- e' solo una
    decisione di QUESTO script di fermare la scalata in n per quella
    combinazione (m, variance, rappresentazione) perche' il tempo per
    istanza ha gia' superato la soglia scelta (di solito la memoria non e'
    ancora esaurita a quel punto: la DP proseguirebbe, solo sempre piu'
    lentamente). Utile per non far durare la calibrazione all'infinito, ma
    NON va confuso con un vero limite di memoria quando si sceglie la
    soglia "n grande" per la campagna finale.

Ogni riga del CSV di dettaglio (--out) e' una singola combinazione
(m, variance, n, representation): il numero di righe e' quindi ~doppio
rispetto a una calibrazione a singola rappresentazione, perche' ogni
istanza viene risolta due volte (una forzando "sparse", una forzando
"dense") finche' entrambe restano attive per quella classe. Il CSV di
riepilogo (--summary-out) aggrega invece per classe (m, variance) con una
riga sola e una raccomandazione leggibile.
"""
import argparse
import sys, os, csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from instances.instance_generator import generate_instance
from solvers.dp_solver import solve_dp, choose_representation
from solvers.heuristic_solver import solve_lpt

grid = {
    2: [50, 100, 200, 400, 800],
    3: [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200],
    5: [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],
}
grid_var = ["low", "high"]
REPRESENTATIONS = ["sparse", "dense"]


def _run_representation(inst, m, representation, memory_limit_bytes):
    return solve_dp(inst.processing_times, m, memory_limit_bytes=memory_limit_bytes,
                     reconstruct_assignment=False, force_representation=representation)


def _build_summary(rows, representations):
    """Aggrega le righe di dettaglio per classe (m, variance).

    Se representations == ["sparse", "dense"] (ordine indifferente, e' il
    default), il comportamento e' quello originale invariato: confronta le
    due rappresentazioni sulla stessa istanza e produce una raccomandazione
    con lo speedup osservato densa/sparsa.

    Se representations contiene una sola rappresentazione (calibrazione
    forzata con --representation sparse|dense), il confronto non e'
    possibile -- l'altra rappresentazione non e' mai stata eseguita in
    questa run -- quindi il riepilogo si limita, per quella sola
    rappresentazione, a fino a che n ha chiuso OPTIMAL e dove si e'
    fermata (memoria o soft cap di tempo), senza colonne di speedup."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[(r["m"], r["variance"])].append(r)

    both = set(representations) == {"sparse", "dense"}
    summary = []
    for (m, var), rs in sorted(by_class.items()):
        by_n = defaultdict(dict)
        for r in rs:
            by_n[r["n"]][r["representation"]] = r
        ns_sorted = sorted(by_n.keys())

        if both:
            last_n_dense_optimal = None      # ultimo n in cui la densa ha chiuso OPTIMAL
            first_n_dense_stopped = None     # primo n in cui la densa si e' fermata (memoria/tempo)
            speedups = []                    # tempo_sparsa / tempo_densa quando entrambe OPTIMAL
            n_dense_slower_than_sparse = 0   # quante volte, a parita' di n, la densa e' stata piu' lenta

            for n in ns_sorted:
                d = by_n[n].get("dense")
                s = by_n[n].get("sparse")
                if d is not None and d["dp_status"] == "OPTIMAL":
                    last_n_dense_optimal = n
                    if s is not None and s["dp_status"] == "OPTIMAL" and d["dp_runtime"] > 0:
                        speedup = s["dp_runtime"] / d["dp_runtime"]
                        speedups.append(speedup)
                        if speedup < 1.0:
                            n_dense_slower_than_sparse += 1
                elif d is not None and d["dp_status"] != "OPTIMAL" and first_n_dense_stopped is None:
                    first_n_dense_stopped = n

            avg_speedup = sum(speedups) / len(speedups) if speedups else None
            min_speedup = min(speedups) if speedups else None

            if last_n_dense_optimal is None:
                recommendation = "usa sempre la sparsa (la densa non e' mai riuscita in questo grid)"
            elif first_n_dense_stopped is None:
                recommendation = (f"usa la densa fino ad almeno n={last_n_dense_optimal} "
                                   f"(limite superiore del grid testato, non ancora fallita)")
            else:
                recommendation = (f"usa la densa per n<={last_n_dense_optimal}, poi passa alla "
                                   f"sparsa (la densa si ferma a n={first_n_dense_stopped})")
            if n_dense_slower_than_sparse > 0:
                recommendation += (f"; attenzione: in {n_dense_slower_than_sparse} classe/i di n "
                                    f"la densa e' risultata PIU' LENTA della sparsa nonostante fosse "
                                    f"eseguibile (overhead numpy su istanze piccole)")

            summary.append(dict(
                m=m, variance=var, dims=m - 1,
                last_n_dense_optimal=last_n_dense_optimal,
                first_n_dense_stopped=first_n_dense_stopped,
                avg_speedup_dense_vs_sparse=round(avg_speedup, 2) if avg_speedup else None,
                min_speedup_dense_vs_sparse=round(min_speedup, 2) if min_speedup else None,
                recommendation=recommendation,
            ))
        else:
            rep = representations[0]
            last_n_optimal = None
            first_n_stopped = None
            for n in ns_sorted:
                r = by_n[n].get(rep)
                if r is not None and r["dp_status"] == "OPTIMAL":
                    last_n_optimal = n
                elif r is not None and r["dp_status"] != "OPTIMAL" and first_n_stopped is None:
                    first_n_stopped = n

            if last_n_optimal is None:
                recommendation = f"rappresentazione '{rep}' forzata: mai chiusa OPTIMAL in questo grid"
            elif first_n_stopped is None:
                recommendation = (f"rappresentazione '{rep}' forzata: OK fino ad almeno n={last_n_optimal} "
                                   f"(limite superiore del grid testato, non ancora fermata)")
            else:
                recommendation = (f"rappresentazione '{rep}' forzata: OK per n<={last_n_optimal}, poi si "
                                   f"ferma a n={first_n_stopped} (nessun confronto disponibile con "
                                   f"l'altra rappresentazione, non eseguita in questa run)")

            summary.append(dict(
                m=m, variance=var, dims=m - 1, representation=rep,
                last_n_optimal=last_n_optimal,
                first_n_stopped=first_n_stopped,
                recommendation=recommendation,
            ))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-limit-gb", type=float, default=16.0,
                     help="deve corrispondere al budget usato nella campagna finale")
    ap.add_argument("--time-soft-cap-s", type=float, default=120.0,
                     help="oltre questo tempo per una singola istanza/rappresentazione, si "
                          "smette di scalare in n per QUELLA rappresentazione (l'altra "
                          "continua finche' non si ferma anche lei) -- vedi "
                          "dp_status=TIME_SOFT_CAP_STOPPED nell'output")
    ap.add_argument("--out", type=str, default="results/calibration_dp_representation.csv",
                     help="CSV di dettaglio, una riga per (m, variance, n, representation)")
    ap.add_argument("--summary-out", type=str,
                     default="results/calibration_dp_representation_summary.csv",
                     help="CSV di riepilogo, una riga per (m, variance) con raccomandazione")
    ap.add_argument("--representation", choices=["sparse", "dense", "both"], default="both",
                     help="rappresentazione da forzare (default: both, comportamento invariato "
                          "-- esegue e confronta entrambe sulla stessa istanza). 'sparse' o "
                          "'dense' eseguono SOLO quella, dimezzando i solve; il riepilogo non "
                          "conterra' lo speedup (vedi _build_summary)")
    args = ap.parse_args()

    memory_limit_bytes = int(args.memory_limit_gb * 1024**3)
    representations = REPRESENTATIONS if args.representation == "both" else [args.representation]

    rows = []
    print(f"memory_limit={args.memory_limit_gb}GB  time_soft_cap={args.time_soft_cap_s}s  "
          f"representations={representations}\n")
    print(f"{'m':>2s} {'var':>5s} {'n':>4s} {'rep':>6s} {'auto':>6s}  "
          f"{'dp_status':>22s} {'dp_runtime':>10s} {'dp_solver_rt':>12s} {'dp_mem_mb':>10s} "
          f"{'dp_states':>12s}")
    for m, ns in grid.items():
        for var in grid_var:
            # stop indipendente per rappresentazione: quando una si ferma
            # (memoria o soft cap di tempo) per questa classe (m, var), si
            # smette di provarla per gli n successivi, ma l'altra continua
            # (se ne e' stata forzata una sola, questo dict ha una sola chiave)
            stopped = {rep: None for rep in representations}
            for n in ns:
                if all(stopped.values()):
                    break
                inst = generate_instance(n=n, m=m, variance=var, seed=1000 + n)
                lpt = solve_lpt(inst.processing_times, m)
                auto_pick = choose_representation(inst.P, m - 1, memory_limit_bytes)

                for rep in representations:
                    if stopped[rep]:
                        continue
                    dp = _run_representation(inst, m, rep, memory_limit_bytes)

                    row_stop_reason = None
                    if dp.status == "OPTIMAL" and dp.runtime > args.time_soft_cap_s:
                        # arrivata all'ottimo comunque: non sovrascriviamo
                        # dp_status, ma segnaliamo che per QUESTA rappresentazione
                        # ci fermiamo qui
                        row_stop_reason = "TIME_SOFT_CAP_STOPPED"
                    elif dp.status == "MEMORY_LIMIT_EXCEEDED":
                        row_stop_reason = "MEMORY_LIMIT_EXCEEDED"
                    elif dp.status == "ERROR":
                        row_stop_reason = "ERROR"
                    if row_stop_reason:
                        stopped[rep] = row_stop_reason

                    solver_rt_str = f"{dp.solver_runtime:.3f}" if dp.solver_runtime is not None else "n/a"
                    mem_str = f"{dp.peak_memory_mb:.1f}" if dp.peak_memory_mb is not None else "n/a"
                    print(f"{m:2d} {var:>5s} {n:4d} {rep:>6s} {auto_pick:>6s}  "
                          f"{dp.status:>22s} {dp.runtime:10.3f} {solver_rt_str:>12s} {mem_str:>10s} "
                          f"{dp.n_states_final:12d}"
                          + (f"   <- stop[{rep}]: {row_stop_reason}" if row_stop_reason else ""))

                    rows.append(dict(m=m, variance=var, n=n, representation=rep,
                                      auto_pick=auto_pick,
                                      dp_status=dp.status,
                                      dp_runtime=round(dp.runtime, 3),
                                      dp_solver_runtime=(round(dp.solver_runtime, 3)
                                                          if dp.solver_runtime is not None else None),
                                      dp_peak_memory_mb=(round(dp.peak_memory_mb, 1)
                                                          if dp.peak_memory_mb is not None else None),
                                      dp_states=dp.n_states_final, dp_cmax=dp.cmax,
                                      lpt_cmax=lpt.cmax, lpt_time_s=round(lpt.runtime, 5),
                                      stop_reason=row_stop_reason))

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDettaglio salvato in {out_path} ({len(rows)} righe)")

    summary = _build_summary(rows, representations)
    summary_path = os.path.join(os.path.dirname(__file__), args.summary_out)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Riepilogo salvato in {summary_path}\n")

    if args.representation == "both":
        print(f"{'m':>2s} {'var':>5s} {'dims':>4s} {'ultimo n densa OK':>18s} "
              f"{'primo n densa ferma':>20s} {'speedup medio':>14s}  raccomandazione")
        for s in summary:
            print(f"{s['m']:2d} {s['variance']:>5s} {s['dims']:4d} "
                  f"{str(s['last_n_dense_optimal']):>18s} "
                  f"{str(s['first_n_dense_stopped']):>20s} "
                  f"{str(s['avg_speedup_dense_vs_sparse']):>14s}  {s['recommendation']}")
    else:
        print(f"{'m':>2s} {'var':>5s} {'dims':>4s} {'rep':>6s} {'ultimo n OK':>12s} "
              f"{'primo n fermata':>16s}  raccomandazione")
        for s in summary:
            print(f"{s['m']:2d} {s['variance']:>5s} {s['dims']:4d} {s['representation']:>6s} "
                  f"{str(s['last_n_optimal']):>12s} "
                  f"{str(s['first_n_stopped']):>16s}  {s['recommendation']}")


if __name__ == "__main__":
    main()
