"""BRAIN: jointly choose one MSU base and its maximum dispatch driving time.

All times are minutes except the input travel matrix (seconds). Rows are origins,
columns are destinations. Outcomes count all modelled ischemic strokes under a
common-IVT scenario; all confirmed/suspected calls contribute to vehicle load.
The outcome curve stays at 15.5% after 270 minutes (an extrapolation assumption).

Inputs: a square pandas DataFrame of directed travel times in seconds, a Series
of annual stroke counts with matching labels (omitted locations have zero demand),
and lists of hospital and optional ambulance-base labels. Every matrix location
is a candidate MSU base. Radii and Parameters use minutes. Location labels should
be strings so policy ties have a consistent ordering. No study data is bundled.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Parameters:
    onset_to_alarm_min: float = 43.0
    msu_dnt_min: float = 22.0
    fixed_response_min: float = 8.0  # Used only when no ambulance bases are supplied.
    conv_scene_min: float = 24.0
    msu_scene_min: float = 30.0      # Includes MSU door-to-needle time.
    hospital_dnt_min: float = 29.0
    handover_min: float = 15.0
    ischemic_fraction: float = 0.86
    minutes_per_year: float = 525_600.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.ischemic_fraction > 1 or self.minutes_per_year == 0:
            raise ValueError("Population fraction must be <= 1 and year length > 0")


DEFAULT = Parameters()


def outcome_percent(ott):
    """Discharge mRS 0-1 (%): linear between knots, horizontal beyond endpoints.

    The 20-270 minute segment approximates Kim et al. (2017), Figure 2B.
    The constant tail is a modelling assumption, not evidence of late IVT efficacy.
    """
    return np.interp(ott, [20, 100, 140, 270], [31, 22.5, 19.5, 15.5])


def validate_locations(travel, locations):
    locations = list(dict.fromkeys(locations))
    missing = [x for x in locations if x not in travel.index or x not in travel.columns]
    if not locations or missing:
        raise ValueError(f"Locations must be nonempty and on both matrix axes; missing: {missing}")
    return locations


@dataclass
class _Inputs:
    travel: pd.DataFrame
    strokes: np.ndarray
    conventional_ott: np.ndarray
    hospital: pd.Series
    hospital_travel: np.ndarray
    params: Parameters


def _prepare(travel, strokes, hospitals, ambulance_bases, params):
    if (not travel.index.is_unique or not travel.columns.is_unique
            or set(travel.index) != set(travel.columns)):
        raise ValueError("Travel matrix needs unique, matching origin/destination labels")
    travel = travel.reindex(columns=travel.index)
    if not np.isfinite(travel.to_numpy()).all() or (travel.to_numpy() < 0).any():
        raise ValueError("Travel times must be finite and nonnegative")
    if (not strokes.index.is_unique or len(strokes.index.difference(travel.index))
            or not np.isfinite(strokes.to_numpy()).all() or (strokes < 0).any()):
        raise ValueError("Stroke counts must be finite, nonnegative and match matrix labels")
    hospitals = validate_locations(travel, hospitals)
    response = params.fixed_response_min
    if ambulance_bases is not None:
        stations = validate_locations(travel, ambulance_bases)
        response = travel.loc[stations].min(axis=0).reindex(travel.index).to_numpy() / 60
    to_hospital = travel[hospitals].min(axis=1).to_numpy() / 60
    conventional = (params.onset_to_alarm_min + response + params.conv_scene_min
                    + to_hospital + params.hospital_dnt_min)
    return _Inputs(travel, strokes.reindex(travel.index).fillna(0).to_numpy(),
                   conventional, travel[hospitals].idxmin(axis=1), to_hospital, params)


def _base_arrays(data, base):
    p = data.params
    outward = data.travel.loc[base].to_numpy() / 60
    ott = p.onset_to_alarm_min + outward + p.msu_dnt_min
    return_leg = data.travel.loc[data.hospital, base].to_numpy() / 60
    service = outward + p.msu_scene_min + data.hospital_travel + p.handover_min + return_leg
    return outward, ott, service


def _evaluate(data, base, radius, multiplier, arrays):
    if not np.isfinite(radius) or radius < 0 or not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("Radius and call multiplier must be finite and nonnegative")
    outward, ott, service = arrays
    dispatch = (outward <= radius) & (ott < data.conventional_ott)
    calls = data.strokes * multiplier
    offered = float(calls[dispatch].sum())
    occupied = float(calls[dispatch] @ service[dispatch])
    load = occupied / data.params.minutes_per_year
    busy = load / (1 + load)  # Erlang loss model: blocked calls use conventional care.
    population = data.strokes * data.params.ischemic_fraction
    conventional = outcome_percent(data.conventional_ott) / 100
    benefit = dispatch * (1 - busy) * (outcome_percent(ott) / 100 - conventional)
    baseline = float(population @ conventional)
    gain = float(population @ benefit)
    result = {
        "base": base, "radius_min": float(radius),
        "expected_outcomes": baseline + gain, "conventional_outcomes": baseline,
        "additional_outcomes": gain, "busy_probability": busy, "offered_load": load,
        "dispatchable_strokes": float(data.strokes[dispatch].sum()),
        "offered_calls": offered, "served_calls": offered * (1 - busy),
        "mean_service_min": occupied / offered if offered else 0.,
        "registered_strokes": float(data.strokes.sum()),
        "modelled_patients": float(population.sum()),
    }
    return result, dispatch, benefit


def evaluate_policy(base, radius, travel, strokes, hospitals, ambulance_bases=None,
                    params=DEFAULT, call_multiplier=1.):
    """Annual outcomes and availability for a specified base/radius policy."""
    data = _prepare(travel, strokes, hospitals, ambulance_bases, params)
    return _evaluate(data, base, radius, call_multiplier, _base_arrays(data, base))[0]


def optimize_policy(travel, strokes, radii, hospitals, ambulance_bases=None,
                    params=DEFAULT, call_multiplier=1.):
    """Return all base/radius policies, best first; ties use radius then label."""
    data = _prepare(travel, strokes, hospitals, ambulance_bases, params)
    radii = sorted(set(radii))
    if not radii or travel.empty:
        raise ValueError("At least one base and radius are required")
    rows = []
    for base in data.travel.index:
        arrays = _base_arrays(data, base)
        rows.extend(_evaluate(data, base, radius, call_multiplier, arrays)[0] for radius in radii)
    return pd.DataFrame(rows).sort_values(
        ["expected_outcomes", "radius_min", "base"], ascending=[False, True, True]
    ).reset_index(drop=True)


def optimize_coverage(travel, strokes, threshold_min=18):
    """MCLP: select the base covering the most confirmed strokes in 18 minutes."""
    travel = travel.reindex(columns=travel.index)
    coverage = (travel.to_numpy() / 60 <= threshold_min) @ strokes.reindex(travel.index).fillna(0).to_numpy()
    return pd.DataFrame({"base": travel.index, "covered_strokes": coverage}).sort_values(
        ["covered_strokes", "base"], ascending=[False, True]
    ).reset_index(drop=True)


def postcode_benefits(base, radius, travel, strokes, hospitals, ambulance_bases=None,
                      params=DEFAULT, call_multiplier=1.):
    """Decompose the policy gain into postal codes; sum equals additional_outcomes."""
    data = _prepare(travel, strokes, hospitals, ambulance_bases, params)
    outward, ott, service = _base_arrays(data, base)
    result, dispatch, benefit = _evaluate(data, base, radius, call_multiplier, (outward, ott, service))
    population = data.strokes * params.ischemic_fraction
    return pd.DataFrame({
        "postcode": data.travel.index, "strokes": data.strokes,
        "modelled_patients": population, "hospital": data.hospital.to_numpy(),
        "conventional_ott_min": data.conventional_ott, "msu_ott_min": ott,
        "outward_min": outward, "service_min": service, "dispatchable": dispatch,
        "conventional_tail": data.conventional_ott > 270,
        "msu_busy_probability": result["busy_probability"],
        "additional_outcomes": population * benefit,
        "benefit_per_100_ischemic_strokes": np.where(population > 0, 100 * benefit, np.nan),
    })
