"""
Euristica semplice per Pm||Cmax: LPT (Longest Processing Time first).

Idea: si ordinano i job in ordine decrescente di tempo di processamento e si
assegnano uno alla volta alla macchina attualmente meno carica (list
scheduling). Trova sempre una soluzione ammissibile in tempo O(n log n).

Garanzia teorica nota (Graham, 1969): LPT ha un rapporto di approssimazione
pari a 4/3 - 1/(3m) rispetto all'ottimo, quindi è un buon termine di
paragone "di base" per DP ed esatto.
"""

from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import List


@dataclass
class HeuristicResult:
    cmax: int
    assignment: List[int]      # assignment[i] = indice macchina (0-based) del job i
    runtime: float              # secondi
    method: str = "LPT"


def solve_lpt(processing_times: List[int], m: int) -> HeuristicResult:
    start = time.perf_counter()
    n = len(processing_times)

    # Ordina i job per tempo di processamento decrescente, mantenendo l'indice originale
    order = sorted(range(n), key=lambda i: processing_times[i], reverse=True)

    # Min-heap sui carichi delle macchine: (carico_corrente, indice_macchina)
    heap = [(0, h) for h in range(m)]
    heapq.heapify(heap)

    assignment = [-1] * n
    for i in order:
        load, h = heapq.heappop(heap)
        assignment[i] = h
        heapq.heappush(heap, (load + processing_times[i], h))

    loads = [0] * m
    for i in range(n):
        loads[assignment[i]] += processing_times[i]
    cmax = max(loads)

    runtime = time.perf_counter() - start
    return HeuristicResult(cmax=cmax, assignment=assignment, runtime=runtime)


if __name__ == "__main__":
    p = [90, 774, 655, 439, 434, 859, 86, 698, 202, 95]
    res = solve_lpt(p, m=3)
    print(res)
    loads = [0, 0, 0]
    for i, h in enumerate(res.assignment):
        loads[h] += p[i]
    print("Carichi macchine:", loads)
