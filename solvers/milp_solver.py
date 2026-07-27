"""
Formulazione a Programmazione Lineare Intera (MILP) per Pm||Cmax,
risolta con Gurobi.

Variabili:
    x[i,h] in {0,1}   job i assegnato alla macchina h
    Cmax   >= 0        makespan

Vincoli:
    sum_h x[i,h] = 1                per ogni job i      (assegnamento unico)
    sum_i p_i * x[i,h] <= Cmax       per ogni macchina h  (definizione makespan)

Obiettivo:
    minimizzare Cmax

Miglioramenti standard (non alterano la formulazione, solo le prestazioni
del solver, dato che le macchine sono identiche e quindi la formulazione
naive ha molta simmetria):
    - lower bound su Cmax:  Cmax >= max_i p_i  e  Cmax >= ceil(sum_i p_i / m)
    - symmetry breaking:    L_1 >= L_2 >= ... >= L_m  (carichi ordinati)

Gestione del time limit: il solver viene fermato dopo `time_limit` secondi.
Se non ha chiuso il gap, si restituisce il miglior incumbent trovato insieme
al gap relativo/assoluto rispetto al best bound.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB


@dataclass
class MilpResult:
    status: str                  # "OPTIMAL", "TIME_LIMIT", "NO_SOLUTION", "INFEASIBLE"
    cmax: Optional[float]        # valore incumbent (None se nessuna soluzione trovata)
    lower_bound: Optional[float] # best bound del solver
    gap_abs: Optional[float]
    gap_rel: Optional[float]
    runtime: float               # secondi (wall clock, misurato da noi)
    solver_runtime: float        # secondi (Model.Runtime riportato da Gurobi)
    assignment: Optional[List[int]] = None  # assignment[i] = macchina (se disponibile)


def solve_milp(processing_times: List[int], m: int, time_limit: float = 120.0,
               symmetry_breaking: bool = True, verbose: bool = False,
               threads: int = 1) -> MilpResult:
    """threads: numero di thread usati da Gurobi (default 1 = single-thread).
    Impostarlo a 1 rende i tempi di calcolo confrontabili tra istanze e
    riproducibili tra run diverse (il B&B multi-thread di Gurobi non e'
    deterministico: stesso modello, stesso time limit, ma cammini di ricerca
    diversi a seconda di come si intrecciano i thread) ed e' anche il
    confronto piu' onesto verso la DP, che gira anch'essa su un solo core."""
    n = len(processing_times)
    p = processing_times
    P = sum(p)

    start = time.perf_counter()

    model = gp.Model("Pm_Cmax")
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    lb_cmax = max(max(p), math.ceil(P / m))
    Cmax = model.addVar(lb=lb_cmax, vtype=GRB.CONTINUOUS, name="Cmax")

    # assegnamento unico
    model.addConstrs((gp.quicksum(x[i, h] for h in range(m)) == 1 for i in range(n)),
                      name="assign")

    # definizione makespan
    loads = []
    for h in range(m):
        load_h = gp.quicksum(p[i] * x[i, h] for i in range(n))
        model.addConstr(load_h <= Cmax, name=f"load_{h}")
        loads.append(load_h)

    # symmetry breaking: L_0 >= L_1 >= ... >= L_{m-1}
    if symmetry_breaking:
        for h in range(m - 1):
            model.addConstr(loads[h] >= loads[h + 1], name=f"sym_{h}")

    model.setObjective(Cmax, GRB.MINIMIZE)
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0   # ridondante quando l'obiettivo è intero, ma esplicita l'intenzione
    model.optimize()

    solver_runtime = model.Runtime
    sol_count = model.SolCount

    assignment = None
    cmax_val = None
    lower_bound = model.ObjBound if model.Status != GRB.INFEASIBLE else None

    if sol_count > 0:
        cmax_val = model.ObjVal
        assignment = [-1] * n
        for i in range(n):
            for h in range(m):
                if x[i, h].X > 0.5:
                    assignment[i] = h
                    break

    if model.Status == GRB.OPTIMAL:
        status = "OPTIMAL"
        gap_abs = cmax_val - lower_bound
        gap_rel = gap_abs / cmax_val if cmax_val else None
    elif model.Status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
        gap_abs, gap_rel = None, None
    elif sol_count > 0:
        status = "TIME_LIMIT"
        gap_abs = cmax_val - lower_bound
        gap_rel = gap_abs / cmax_val if cmax_val else None
    else:
        status = "NO_SOLUTION"
        gap_abs, gap_rel = None, None

    runtime = time.perf_counter() - start
    model.dispose()

    return MilpResult(status=status, cmax=cmax_val, lower_bound=lower_bound,
                       gap_abs=gap_abs, gap_rel=gap_rel, runtime=runtime,
                       solver_runtime=solver_runtime, assignment=assignment)


if __name__ == "__main__":
    p = [90, 774, 655, 439, 434, 859, 86, 698, 202, 95]
    res = solve_milp(p, m=3, time_limit=30, verbose=False)
    print(res)
