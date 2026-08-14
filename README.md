# Mobile Stroke Unit placement — optimization models

A Mobile Stroke Unit (MSU) is a specialized ambulance with an on-board CT scanner and clinical staff, so stroke diagnosis and treatment (e.g. thrombolysis) can start *before* the patient reaches a hospital. Since a region typically operates only one or a few MSUs, where to base them matters: every minute earlier treatment starts improves the patient's chance of recovering without disability.

This repository is a small example of the two optimization
models we used in a study on MSU placement in the Oslo region (Norway). It contains **only the mathematical models** and no study data. A tiny synthetic example is included so everything runs out of the box.

- [`msu_models.py`](msu_models.py) — the two models, built with
  [Pyomo](https://www.pyomo.org) so you can use **any solver you like** (HiGHS, CBC, GLPK, Gurobi, CPLEX, etc.)
- [`example.ipynb`](example.ipynb) — a walkthrough on a synthetic instance, with a map of the resulting placements

## The models
Both models are variants of the **Maximal Covering Location Problem (MCLP)**: choose `p` base locations so that as many stroke patients as possible can be reached within a time threshold.

**Sets and data**

| Symbol | Meaning |
|---|---|
| `I` | candidate MSU base locations |
| `J` | demand points (e.g. postal codes) |
| `w_j` | demand at point `j` (e.g. strokes per year) |
| `t_ij` | travel time from base `i` to demand point `j` |
| `D` | coverage time threshold (18 minutes in our study) |
| `p` | number of MSUs to place |
| `β_ij` | 1 if point `j` is reachable from base `i` via an ambulance rendezvous (rendezvous model only) |

**Variables**: `x_i = 1` if an MSU is based at `i`; `y_j = 1` if demand point `j` is covered.

```
maximize    Σ_j  w_j · y_j                 (covered demand)

subject to  Σ_i  x_i = p                   (place exactly p MSUs)

            y_j ≤ Σ_{i : t_ij ≤ D} x_i     (direct coverage)
                  + Σ_i β_ij · x_i         (rendezvous coverage — omit for the plain model)

            x_i, y_j ∈ {0, 1}
```

### Model 1: without rendezvous

A demand point is covered if an open MSU base can reach it within the threshold `D`. This is the classic MCLP — set all `β_ij = 0` above.

### Model 2: with rendezvous

An MSU can effectively reach further if a regular ambulance (usually closer to the patient) picks the patient up and drives *towards* the MSU while the MSU drives towards them, transferring the patient where they meet. The model is identical except for the extra `β` term in the coverage constraint.

`β_ij` is precomputed from travel times and the ambulance base locations:

```
β_ij = 1   iff   t_amb(j) + v + s + t_ij ≤ 2 · (D + v)
                 and t_amb(j) ≤ T_amb
```

where `t_amb(j)` is the travel time from the closest ambulance base to `j`, `v` the on-scene treatment time the ambulance performs anyway (22 min in the study), `s` the patient transfer time at the meeting point (2 min), and `T_amb` the maximum acceptable ambulance response time (16 min). Intuitively: the ambulance and MSU drive towards each other, so the patient can be inside the MSU no later than an "MSU at the door at time `D`" scenario would allow.

## Installation

```bash
pip install -r requirements.txt
```

This installs Pyomo plus the open-source [HiGHS](https://highs.dev) solver (`highspy`), which is enough to run everything. Jupyter and matplotlib are only needed for the notebook.

## Quickstart

```python
from msu_models import build_msu_model, compute_rendezvous_coverage, solve

# Your data: any hashable location labels, travel times in seconds.
demand = {"A": 12, "B": 5, "C": 8, "D": 2}
travel_times = {("A", "A"): 0, ("A", "B"): 600, ...}  # (from, to) -> seconds

# Model 1: plain coverage.
model = build_msu_model(demand, travel_times, time_threshold=18 * 60, n_msus=1)
result = solve(model)
print(result.chosen_locations, result.covered_demand)

# Model 2: with ambulance rendezvous.
beta = compute_rendezvous_coverage(
    travel_times, ambulance_bases=["B"], demand_points=demand
)
model_rv = build_msu_model(demand, travel_times, rendezvous_coverage=beta)
result_rv = solve(model_rv)
```

Scenario analysis ("what if the MSU must be based at hospital X?") works via `fixed_locations=["X"]`.

## Using a different solver

Pass any solver Pyomo knows to `solve`:

```python
solve(model)                      # HiGHS (default, pip-installable)
solve(model, solver="cbc")        # COIN-OR CBC   (e.g. `conda install coin-or-cbc`)
solve(model, solver="glpk")       # GLPK          (e.g. `brew install glpk`)
solve(model, solver="gurobi")     # Gurobi        (license required)
```

Or skip the helper entirely and use the returned Pyomo model with your own solver setup: `build_msu_model` returns an ordinary `pyo.ConcreteModel`.

## Citation

These models accompany the following paper — please cite it if you use them:

> Markhorst, B., Jagtenberg, C., Hov, M. R., van der Mei, R., & Larsen, K. (2025).
> Optimizing mobile stroke unit deployment: A strategic case study in the greater
> Oslo area. *European Stroke Journal*, 10(4), 1438–1444.
> [doi:10.1177/23969873251329862](https://doi.org/10.1177/23969873251329862)

```bibtex
@article{markhorst2025optimizing,
  title={Optimizing mobile stroke unit deployment: A strategic case study in the greater Oslo area},
  author={Markhorst, Berend and Jagtenberg, Caroline and Hov, Maren Ranhoff and van der Mei, Rob and Larsen, Karianne},
  journal={European Stroke Journal},
  volume={10},
  number={4},
  pages={1438--1444},
  year={2025},
  doi={10.1177/23969873251329862},
  publisher={SAGE Publications Sage UK: London, England}
}
```

## License

[MIT](LICENSE) — you are free to use, modify, and redistribute this code, including commercially, as long as the copyright notice is kept. Please do consider a citation in whatever context you use this code.
