"""
Generatore di istanze per il problema Pm||Cmax
(macchine parallele identiche, minimizzazione del makespan).

Parametri di generazione (come da specifica del progetto):
  - n: numero di job
  - m: numero di macchine
  - variance: 'low' o 'high'
        low  -> tempi concentrati intorno a una media (Normale troncata su [1,100])
        high -> tempi distribuiti sull'intero range [1,1000] (job molto corti
                 e molto lunghi possono comparire nella stessa istanza)

Ogni istanza viene rappresentata come un dizionario serializzabile in JSON:
{
    "id": str,
    "n": int,
    "m": int,
    "variance": "low"/"high",
    "seed": int,
    "processing_times": [p_1, ..., p_n]
}
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List
import numpy as np


@dataclass
class Instance:
    id: str
    n: int
    m: int
    variance: str
    seed: int
    processing_times: List[int]

    @property
    def P(self) -> int:
        return int(sum(self.processing_times))

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path) as f:
            d = json.load(f)
        return Instance(**d)


def _draw_processing_times(n: int, variance: str, rng: np.random.Generator,
                            low_mean: float = 50.0, low_std: float = 10.0,
                            high_low: int = 1, high_high: int = 1000) -> List[int]:
    if variance == "low":
        # Normale troncata su [1,100], concentrata intorno a low_mean
        raw = rng.normal(loc=low_mean, scale=low_std, size=n)
        clipped = np.clip(np.round(raw), 1, 100).astype(int)
        return clipped.tolist()
    elif variance == "high":
        # Uniforme sull'intero range: convivono job cortissimi e lunghissimi
        raw = rng.integers(low=high_low, high=high_high + 1, size=n)
        return raw.tolist()
    else:
        raise ValueError(f"variance deve essere 'low' o 'high', ricevuto: {variance}")


def generate_instance(n: int, m: int, variance: str, seed: int,
                       id_prefix: str = "inst") -> Instance:
    rng = np.random.default_rng(seed)
    p = _draw_processing_times(n, variance, rng)
    inst_id = f"{id_prefix}_n{n}_m{m}_{variance}_seed{seed}"
    return Instance(id=inst_id, n=n, m=m, variance=variance, seed=seed,
                     processing_times=p)


def generate_instance_custom_range(n: int, m: int, seed: int, p_low: int, p_high: int,
                                    id_prefix: str = "inst_customrange") -> Instance:
    """Istanza con tempi di processamento uniformi in [p_low, p_high], al
    di fuori delle due classi fisse 'low'/'high' usate nella campagna
    sperimentale principale (che restano invariate per riproducibilita').
    Pensata per esplorazioni ad hoc -- es. calibrazione della DP con tempi
    "pochi unita'" (p_low/p_high piccoli) e molte macchine, vedi
    calibrate_dp_low_p_many_m.py -- dove serve controllare direttamente il
    range dei tempi invece di usare le medie/varianze fissate di
    generate_instance(). Il campo `variance` dell'Instance risultante viene
    valorizzato con una stringa descrittiva (non 'low'/'high') solo a scopo
    di tracciabilita' nei CSV di output.
    """
    if p_low < 1 or p_high < p_low:
        raise ValueError(f"range non valido: p_low={p_low}, p_high={p_high} (serve 1 <= p_low <= p_high)")
    rng = np.random.default_rng(seed)
    p = rng.integers(low=p_low, high=p_high + 1, size=n).tolist()
    inst_id = f"{id_prefix}_n{n}_m{m}_p{p_low}-{p_high}_seed{seed}"
    return Instance(id=inst_id, n=n, m=m, variance=f"custom[{p_low},{p_high}]", seed=seed,
                     processing_times=p)


def generate_batch(n_values: List[int], m_values: List[int],
                    variance_values: List[str], n_instances_per_class: int,
                    out_dir: str, base_seed: int = 0) -> List[str]:
    """
    Genera n_instances_per_class istanze pseudocasuali per ogni combinazione
    (n, m, variance) e le salva come JSON in out_dir.
    Ritorna la lista dei path creati.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    seed_counter = base_seed
    for n in n_values:
        for m in m_values:
            for variance in variance_values:
                for k in range(n_instances_per_class):
                    seed = seed_counter
                    seed_counter += 1
                    inst = generate_instance(n, m, variance, seed)
                    path = os.path.join(out_dir, f"{inst.id}_{k}.json")
                    inst.save(path)
                    paths.append(path)
    return paths


if __name__ == "__main__":
    # Piccolo self-test
    inst = generate_instance(n=10, m=3, variance="low", seed=42)
    print(inst)
    print("P totale:", inst.P)
    inst2 = generate_instance(n=10, m=3, variance="high", seed=42)
    print(inst2)
    print("P totale:", inst2.P)
