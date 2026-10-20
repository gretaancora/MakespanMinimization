# Makespan Minimization on Identical Parallel Machines

## Executive Summary

This repository studies the identical parallel-machine scheduling problem commonly written as **P\|\|Cmax**. Given a set of jobs with processing times and a fixed number of identical machines, the objective is to assign every job to exactly one machine while minimizing the makespan, i.e. the largest total processing load assigned to any machine.

The project combines exact optimization, dynamic programming, and a scheduling heuristic. It also includes instance generation, calibration scripts, and an experimental harness for comparing solution quality, running time, and memory use across different instance sizes and processing-time distributions.

## Technical Scope & Key Findings

### Problem model

- **Jobs (`n`)**: each job has a positive processing time `p[i]`.
- **Machines (`m`)**: identical parallel machines, with no job preemption represented in the model.
- **Objective (`Cmax`)**: minimize the maximum machine load.
- **Evaluation dimensions**: number of jobs, number of machines, processing-time variance, solver status, runtime, memory consumption, and solution quality where available.

### Implemented solution methods

| Method | Implementation | Purpose |
|---|---|---|
| Mixed-Integer Linear Programming (MILP) | `solvers/milp_solver.py` | Exact formulation solved with Gurobi. Includes a makespan lower bound and optional symmetry-breaking constraints. |
| Dynamic programming (DP) | `solvers/dp_solver.py` | Pseudo-polynomial method with automatic selection between dense NumPy and sparse tuple-set representations, plus memory-limit handling. |
| Longest Processing Time first (LPT) | `solvers/heuristic_solver.py` | Greedy baseline: order jobs by decreasing processing time and repeatedly assign the next job to the least-loaded machine. |

The methods have different operational trade-offs: MILP can provide an optimality certificate or a best bound, DP can be limited by state-space growth and memory, and LPT produces a fast feasible baseline but does not generally certify optimality.

### Instance generation and experiments

The instance generator supports two processing-time classes:

- `low`: processing times sampled around a central value and clipped to the range `[1, 100]`.
- `high`: processing times sampled over the wider range `[1, 1000]`.

`experiments/experiment_runner.py` runs the selected methods over a batch of stored or generated instances and writes the measurements to CSV. The `calibrate_dp_*.py` scripts investigate practical DP limits, representation choices, and the impact of increasing the number of jobs or machines under controlled parameter ranges.

## Repository Contents

| Path | Role |
|---|---|
| `solvers/` | MILP, dynamic-programming, and LPT solver implementations. |
| `instances/` | Instance data and generation utilities. |
| `experiments/` | Batch experiment runner and evaluation pipeline. |
| `analysis/` | Supporting analysis material. |
| `results/` | Location for experiment and calibration outputs. |
| `calibrate_dp_lpt.py` | Compares the dense and sparse DP representations and calibrates their practical limits. |
| `calibrate_dp_low_p_many_m.py` | Studies scaling as the machine count increases while keeping processing times small. |
| `calibrate_dp_low_p_scaling_n.py` | Calibration focused on how DP behavior changes as the job count grows. |
| `calibrate_dp_low_p_sweep_n.py` | Sweep-style calibration for low processing-time ranges. |
| `requirements.txt` | Python package requirements. |
| `Presentazione_AMOD.pdf` | Project presentation. |

## Requirements & Environment

- Python 3.10 or newer is recommended for the current type-annotated code.
- Packages listed in `requirements.txt`: NumPy, pandas, Gurobi's Python interface (`gurobipy`), and Matplotlib.
- A working Gurobi installation and an applicable license are required to run the MILP method. Check Gurobi's installation and licensing instructions for your environment.

Create an isolated Python environment and install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of the `source` command.

## Running the Experiments

From the repository root, a representative generated experiment campaign is:

```bash
python experiments/experiment_runner.py \
  --generate \
  --m 2 3 5 \
  --n 20 50 100 \
  --variance low high \
  --instances-per-class 10 \
  --time-limit 120 \
  --memory-limit-gb 2 \
  --out results/results.csv
```

This command generates a batch of instances and compares the available methods with a 120-second time limit and a 2 GB DP memory budget. The precise workload depends on the selected values of `m`, `n`, variance, and instances per class; a full campaign can take a significant amount of time.

To run the DP calibration script and compare its two representations:

```bash
python calibrate_dp_lpt.py \
  --memory-limit-gb 16 \
  --time-soft-cap-s 120 \
  --representation both
```

For supported options and defaults, use the scripts' built-in help, for example:

```bash
python experiments/experiment_runner.py --help
python calibrate_dp_lpt.py --help
```

The runner can also load existing JSON instances from its configured instances directory by omitting `--generate`. The main experiment output is CSV, with status and performance fields for the methods that were run.

## Interpreting Results

- Treat `OPTIMAL` as an optimality result only when reported by the solver for the full instance.
- A MILP run that reaches its time limit may still return an incumbent and a lower bound; these are useful, but they are not an optimality certificate.
- DP can stop when its memory budget is exceeded. A partial DP value should not be interpreted as a feasible makespan for all jobs unless the result explicitly reports a valid complete solution.
- LPT is a feasible heuristic baseline. Compare its makespan against a proven optimum or the best available bound when one exists.
- Benchmark results depend on hardware, Python/NumPy versions, solver settings, the Gurobi version and license, and the configured time and memory limits.

## Academic Context

The repository includes `Presentazione_AMOD.pdf` and supporting analysis files. Consult those materials for the project-specific formulation and experimental discussion.
