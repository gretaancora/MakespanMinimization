"""Brute force esatto (solo per validazione su istanze piccolissime, n<=~12)."""
from itertools import product


def solve_brute_force(processing_times, m):
    n = len(processing_times)
    best = None
    for assignment in product(range(m), repeat=n):
        loads = [0] * m
        for i, h in enumerate(assignment):
            loads[h] += processing_times[i]
        cmax = max(loads)
        if best is None or cmax < best:
            best = cmax
    return best
