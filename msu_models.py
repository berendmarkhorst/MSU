"""Optimization models for Mobile Stroke Unit (MSU) placement.

A Mobile Stroke Unit is a specialized ambulance with an on-board CT scanner
and clinical staff, so stroke treatment can start before the patient reaches
a hospital. Because a region typically operates only one or a few MSUs, the
question of *where to base them* is critical: every minute earlier treatment
starts improves the patient's odds of recovering without disability.

This module contains the two location models used in our study on MSU
placement in the Oslo region (Norway), stripped of all study data:

1. **Without rendezvous** — a classic Maximal Covering Location Problem
   (MCLP): a patient is covered if an MSU base is within the time threshold.
2. **With rendezvous** — the same model with one extra term: a patient may
   also be covered if a regular ambulance can pick them up and meet the MSU
   halfway ("rendezvous"), which extends the MSU's effective reach.

If you use these models, please cite:

    Markhorst, B., Jagtenberg, C., Hov, M. R., van der Mei, R., & Larsen, K.
    (2025). Optimizing mobile stroke unit deployment: A strategic case study
    in the greater Oslo area. European Stroke Journal, 10(4), 1438-1444.
    https://doi.org/10.1177/23969873251329862

Mathematical formulation
------------------------
Sets and data:

- ``I``: candidate MSU base locations
- ``J``: demand points (e.g. postal codes), with demand ``w_j`` (number of
  strokes) and travel time ``t_ij`` from base ``i`` to point ``j``
- ``D``: coverage time threshold (18 minutes in our study)
- ``p``: number of MSUs to place
- ``beta_ij`` (rendezvous model only): 1 if a patient at ``j`` can be
  transferred to an MSU based at ``i`` within the threshold via an
  ambulance rendezvous, 0 otherwise (see ``compute_rendezvous_coverage``)

Variables:

- ``x_i = 1`` if an MSU is based at location ``i``
- ``y_j = 1`` if demand point ``j`` is covered

Model::

    maximize    sum_j w_j * y_j
    subject to  sum_i x_i = p
                y_j <= sum_{i : t_ij <= D} x_i  +  sum_i beta_ij * x_i
                x_i, y_j binary

The rendezvous model differs *only* by the ``beta`` term in the coverage
constraint; drop it (``beta_ij = 0``) and you have the plain MCLP.

The models are built with `Pyomo <https://www.pyomo.org>`_, so any solver
Pyomo supports can be used (HiGHS, CBC, GLPK, Gurobi, CPLEX, ...).

Quickstart
----------
>>> demand = {"A": 10, "B": 5, "C": 8}
>>> travel_times = {(i, j): 0 if i == j else 600
...                 for i in "ABC" for j in "ABC"}
>>> model = build_msu_model(demand, travel_times, time_threshold=18 * 60)
>>> result = solve(model)          # doctest: +SKIP
>>> result.chosen_locations       # doctest: +SKIP
['A']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Optional

import pyomo.environ as pyo

Location = Hashable  # anything usable as a dict key: postal code strings, ints, ...

#: Default coverage threshold used in the study: 18 minutes, in seconds.
DEFAULT_TIME_THRESHOLD = 18 * 60


def build_msu_model(
    demand: Mapping[Location, float],
    travel_times: Mapping[tuple[Location, Location], float],
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    n_msus: int = 1,
    candidates: Optional[Iterable[Location]] = None,
    fixed_locations: Optional[Iterable[Location]] = None,
    rendezvous_coverage: Optional[Mapping[tuple[Location, Location], int]] = None,
) -> pyo.ConcreteModel:
    """Build the MSU placement model (a Maximal Covering Location Problem).

    Parameters
    ----------
    demand:
        Maps each demand point ``j`` to its demand ``w_j``, e.g. the number
        of (suspected) strokes per year in that postal code.
    travel_times:
        Maps ``(i, j)`` to the travel time from candidate base ``i`` to
        demand point ``j``. Use the same time unit as ``time_threshold``
        (the defaults assume seconds). Must contain an entry for every
        candidate/demand-point pair.
    time_threshold:
        A demand point is covered by a base if the travel time is at most
        this threshold (``D`` in the formulation). Default: 18 minutes.
    n_msus:
        Number of MSUs to place (``p`` in the formulation).
    candidates:
        Candidate base locations ``I``. Defaults to the demand points
        themselves, as in our study; pass e.g. a list of hospital or
        ambulance-base locations to restrict the choice.
    fixed_locations:
        Locations forced to receive an MSU (``x_i = 1``). Useful for
        scenario analysis: "how many patients do we cover if the MSU must
        be based at hospital X?". Must be a subset of ``candidates`` and at
        most ``n_msus`` locations.
    rendezvous_coverage:
        The ``beta_ij`` values as a mapping from ``(i, j)`` to 0/1; pairs
        that are absent count as 0. Omit it (the default) for the model
        *without* rendezvous. See :func:`compute_rendezvous_coverage`.

    Returns
    -------
    pyo.ConcreteModel
        The model, ready to be passed to :func:`solve` (or to any Pyomo
        solver directly).
    """
    demand_points = list(demand)
    candidates = list(candidates) if candidates is not None else list(demand_points)
    beta = rendezvous_coverage if rendezvous_coverage is not None else {}

    model = pyo.ConcreteModel("msu_placement")
    model.I = pyo.Set(initialize=candidates, doc="candidate MSU base locations")
    model.J = pyo.Set(initialize=demand_points, doc="demand points")

    model.x = pyo.Var(model.I, domain=pyo.Binary, doc="1 if an MSU is based at i")
    model.y = pyo.Var(model.J, domain=pyo.Binary, doc="1 if demand point j is covered")

    model.total_covered_demand = pyo.Objective(
        expr=sum(demand[j] * model.y[j] for j in model.J),
        sense=pyo.maximize,
        doc="total demand at covered points",
    )

    model.number_of_bases = pyo.Constraint(
        expr=sum(model.x[i] for i in model.I) == n_msus,
        doc="place exactly p MSUs",
    )

    def coverage_rule(model: pyo.ConcreteModel, j: Location) -> pyo.Expression:
        # y_j <= sum_{i: t_ij <= D} x_i + sum_i beta_ij x_i
        # A point is covered only if some open base reaches it directly
        # within the threshold, or (rendezvous model) via an ambulance
        # meeting the MSU en route.
        return model.y[j] <= (
            sum(model.x[i] for i in model.I if travel_times[i, j] <= time_threshold)
            + sum(beta.get((i, j), 0) * model.x[i] for i in model.I)
        )

    model.coverage = pyo.Constraint(model.J, rule=coverage_rule)

    if fixed_locations:
        fixed = list(fixed_locations)
        unknown = [i for i in fixed if i not in set(candidates)]
        if unknown:
            raise ValueError(f"fixed_locations not among candidates: {unknown}")
        model.fixed_locations = pyo.Constraint(
            fixed,
            rule=lambda model, i: model.x[i] == 1,
            doc="locations forced to receive an MSU",
        )

    return model


def build_msu_rendezvous_model(
    demand: Mapping[Location, float],
    travel_times: Mapping[tuple[Location, Location], float],
    rendezvous_coverage: Mapping[tuple[Location, Location], int],
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    n_msus: int = 1,
    candidates: Optional[Iterable[Location]] = None,
    fixed_locations: Optional[Iterable[Location]] = None,
) -> pyo.ConcreteModel:
    """Build the MSU placement model *with* ambulance rendezvous.

    This is :func:`build_msu_model` with a required ``rendezvous_coverage``
    argument — the two models differ only by the ``beta`` term in the
    coverage constraint. See that function for the parameter descriptions
    and :func:`compute_rendezvous_coverage` for how to obtain the ``beta``
    values.
    """
    return build_msu_model(
        demand=demand,
        travel_times=travel_times,
        time_threshold=time_threshold,
        n_msus=n_msus,
        candidates=candidates,
        fixed_locations=fixed_locations,
        rendezvous_coverage=rendezvous_coverage,
    )


def compute_rendezvous_coverage(
    travel_times: Mapping[tuple[Location, Location], float],
    ambulance_bases: Iterable[Location],
    demand_points: Iterable[Location],
    candidates: Optional[Iterable[Location]] = None,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    treatment_time: float = 22 * 60,
    transfer_time: float = 2 * 60,
    max_ambulance_time: float = 16 * 60,
) -> dict[tuple[Location, Location], int]:
    """Compute the rendezvous coverage indicators ``beta_ij``.

    The rendezvous idea: instead of the MSU driving all the way to the
    patient, a regular ambulance (which is usually closer) picks the
    patient up and drives *towards* the MSU while the MSU drives towards
    them; the patient is transferred where they meet. This extends the
    MSU's effective range beyond the plain time threshold.

    ``beta_ij = 1`` when a patient at demand point ``j`` can be inside an
    MSU based at ``i`` "in time", accounting for the on-scene treatment
    the ambulance performs anyway. With ``t_amb(j)`` the travel time from
    the closest ambulance base to ``j``, ``v`` the on-scene treatment
    time, ``s`` the patient transfer (handover) time and ``D`` the
    threshold, the condition from the study is::

        beta_ij = 1  iff  t_amb(j) + v + s + t_ij <= 2 * (D + v)
                          and t_amb(j) <= max_ambulance_time

    The left inequality says the ambulance (after reaching the patient and
    treating on scene) and the MSU, driving towards each other, can meet
    no later than an MSU-at-the-door arrival at time ``D + v`` would allow;
    the right one requires an ambulance to reach the patient reasonably
    fast in the first place.

    Parameters
    ----------
    travel_times:
        Maps ``(a, b)`` to the travel time between locations ``a`` and
        ``b``. Must cover ambulance-base/demand-point pairs as well as
        candidate/demand-point pairs.
    ambulance_bases:
        Locations of the regular ambulance bases.
    demand_points:
        The demand points ``J``.
    candidates:
        Candidate MSU base locations ``I``; defaults to the demand points.
    time_threshold:
        Coverage threshold ``D``. Default: 18 minutes, in seconds.
    treatment_time:
        On-scene treatment time ``v`` the ambulance spends with the
        patient before departing. Default: 22 minutes.
    transfer_time:
        Time ``s`` to move the patient from the ambulance into the MSU at
        the meeting point. Default: 2 minutes.
    max_ambulance_time:
        Maximum time within which some ambulance must be able to reach the
        patient for a rendezvous to be considered. Default: 16 minutes.

    Returns
    -------
    dict
        ``{(i, j): 0 or 1}`` for every candidate ``i`` and demand point
        ``j``, ready to be passed to ``rendezvous_coverage=`` of the model
        builders.
    """
    ambulance_bases = list(ambulance_bases)
    demand_points = list(demand_points)
    candidates = list(candidates) if candidates is not None else list(demand_points)
    if not ambulance_bases:
        raise ValueError("ambulance_bases must not be empty")

    # Travel time from the closest ambulance base to each demand point.
    closest_ambulance_time = {
        j: min(travel_times[a, j] for a in ambulance_bases) for j in demand_points
    }

    beta = {}
    for j in demand_points:
        t_amb = closest_ambulance_time[j]
        for i in candidates:
            reachable = (
                t_amb + treatment_time + transfer_time + travel_times[i, j]
                <= 2 * (time_threshold + treatment_time)
                and t_amb <= max_ambulance_time
            )
            beta[i, j] = int(reachable)
    return beta


@dataclass
class SolveResult:
    """Solution of an MSU placement model.

    Attributes
    ----------
    chosen_locations:
        The locations selected as MSU bases (the ``i`` with ``x_i = 1``).
    covered_points:
        The demand points that are covered (the ``j`` with ``y_j = 1``).
    covered_demand:
        Total demand at the covered points (the objective value), e.g. the
        number of stroke patients reachable within the threshold.
    """

    chosen_locations: list[Location]
    covered_points: list[Location]
    covered_demand: float


def solve(
    model: pyo.ConcreteModel,
    solver: str = "appsi_highs",
    solver_options: Optional[Mapping[str, object]] = None,
    tee: bool = False,
) -> SolveResult:
    """Solve an MSU placement model with any Pyomo-supported solver.

    Parameters
    ----------
    model:
        A model from :func:`build_msu_model` or
        :func:`build_msu_rendezvous_model`.
    solver:
        Pyomo solver name. The default, ``"appsi_highs"``, uses the
        open-source HiGHS solver (``pip install highspy``). Other options
        include ``"cbc"``, ``"glpk"``, ``"gurobi"``, and ``"cplex"`` —
        anything registered with ``pyo.SolverFactory`` works.
    solver_options:
        Solver-specific options, e.g. ``{"time_limit": 60}``.
    tee:
        If True, print the solver log.

    Returns
    -------
    SolveResult
        The chosen base locations, covered demand points, and total
        covered demand.
    """
    opt = pyo.SolverFactory(solver)
    if solver_options:
        for key, value in solver_options.items():
            opt.options[key] = value

    results = opt.solve(model, tee=tee)
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find an optimal solution: {condition}")

    return SolveResult(
        chosen_locations=[i for i in model.I if pyo.value(model.x[i]) > 0.5],
        covered_points=[j for j in model.J if pyo.value(model.y[j]) > 0.5],
        covered_demand=pyo.value(model.total_covered_demand),
    )
