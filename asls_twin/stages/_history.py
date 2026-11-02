"""Shared causal history weighting for Stage 1c and Stage 3 substages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .. import config
from ..calculations import add_orm, regression_features
from ..types import ORM
from . import _residual_cache

# Use the same block ordering as ORM throughout the weighted sums. Cross-plane
# blocks are carried through as well as the horizontal and vertical diagonals.
BLOCKS = ("Rxx", "Ryx", "Rxy", "Ryy")

# the two bandwidths control how quickly weight falls with ID and current distance after those distances are normalized.
@dataclass(frozen=True)
class Weighting:
    half_life_days: float = 90.0
    id_bandwidth: float = 1.0
    current_bandwidth: float = 1.0
    span_floor_fraction: float = 0.01
    span_ceiling_fraction: float | None = None
    feature_scale_floor: float = 1e-9
    current_scale_floor_A: float = 0.001
    sum_id_distances: bool = False

    def check(self) -> "Weighting":
        values = (self.half_life_days, self.id_bandwidth, self.current_bandwidth,
                  self.span_floor_fraction, self.feature_scale_floor,
                  self.current_scale_floor_A)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("history weighting parameters must be finite and positive")
        if self.span_ceiling_fraction is not None and (
            not np.isfinite(self.span_ceiling_fraction)
            or self.span_ceiling_fraction < self.span_floor_fraction
        ):
            raise ValueError("history span ceiling must be finite and at least the floor")
        return self


def _timestamp(item) -> datetime:
    return item.timestamp

# Only use measurements  earlier than prediction time.
def eligible_history(history, *, target_time: datetime | None, require_ids: bool):
    target = target_time or datetime.now()
    eligible = [item for item in history if _timestamp(item) < target]
    if require_ids:
        eligible = [item for item in eligible if item.id_state is not None]
    return sorted(eligible, key=_timestamp)


def history_weights(history, state, current_A, target_time, policy=Weighting()):
    """Weight earlier records by age, transformed ID distance, and current."""
    policy.check()
    if not history:
        raise ValueError("history weighting requires at least one record")
    target = target_time or datetime.now()
    if current_A is None or not np.isfinite(current_A) or not 0 < current_A <= 1:
        raise ValueError("history weighting requires beam_current_A in (0, 1]")
    if any(item.timestamp >= target or item.id_state is None for item in history):
        raise ValueError("history weighting requires earlier ID-associated records")
    # Every selected record needs a valid recorded current for this weighting.
    currents = np.asarray([item.beam_current_A for item in history], float)
    if not np.isfinite(currents).all() or np.any((currents <= 0) | (currents > 1)):
        raise ValueError("history weighting requires recorded historical beam currents")

    # Compare IDs in the same transformed coordinates used by the regression
    features = np.asarray(
        [regression_features(dict(item.id_state)) for item in history]
    )
    query = regression_features(state)
    lows = {name: limits[0] for name, limits in config.ID_READBACK_RANGES.items()}
    highs = {name: limits[1] for name, limits in config.ID_READBACK_RANGES.items()}
    span = np.abs(regression_features(highs) - regression_features(lows))
    scale = np.maximum(features.std(axis=0), np.maximum(
        policy.span_floor_fraction * span, policy.feature_scale_floor))
    if policy.span_ceiling_fraction is not None:
        scale = np.minimum(scale, np.maximum(
            policy.span_ceiling_fraction * span, policy.feature_scale_floor))
    distance = ((features - query) / scale) ** 2
    id_distance2 = (np.sum(distance, axis=1) if policy.sum_id_distances
                    else np.mean(distance, axis=1))
    current_scale = max(float(currents.std()), policy.current_scale_floor_A)
    current_distance2 = ((currents - current_A) / current_scale) ** 2
    age_days = np.asarray(
        [(target - item.timestamp).total_seconds() / 86400.0 for item in history]
    )
    log_weights = (
        -np.log(2.0) * age_days / policy.half_life_days
        - 0.5 * id_distance2 / policy.id_bandwidth**2
        - 0.5 * current_distance2 / policy.current_bandwidth**2
    )
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()

# Average measured or residual ORMs using non-negative weights. The matrices
# must be in nm/A
def weighted_orm(values, weights):
    weights = np.asarray(weights, float)
    if (
        not values
        or weights.shape != (len(values),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("invalid ORM weights")
    for value in values:
        value.check()
        if value.units != "nm/A":
            raise ValueError("historical ORMs must use nm/A")
    # Normalize the supplied weights
    weights = weights / weights.sum()
    return ORM(
        *(
            sum(
                (weight * getattr(value, block) for weight, value in zip(weights, values)),
                np.zeros_like(getattr(values[0], block)),
            )
            for block in BLOCKS
        ),
        "nm/A",
    ).check()


def _cached_residual(baseline, item, *, workers):
    """Return measured minus modelled for one record, reusing the disk cache.

    The residual depends only on the measurement and the baseline model, both of
    which are fixed, so it is computed at most once. A cache fault is never
    allowed to stop a prediction; it just costs the rebuild.
    """
    store = None
    key = None
    try:
        store = _residual_cache.store()
        if store is not None:
            key = _residual_cache.entry_key(baseline.__module__, item)
            hit = store.get(key)
            if hit is not None:
                return hit
    except Exception:
        store = None
    # Rebuild the physics at the historical ID state and current
    modeled = baseline(
        dict(item.id_state),
        [],
        workers=workers,
        beam_current_A=item.beam_current_A,
        target_time=item.timestamp,
        machine_state=None,
    )
    residual = add_orm(item.orm, modeled, scale=-1.0)
    if store is not None and key is not None:
        try:
            store.put(key, residual)
            store.flush()
        except Exception:
            pass
    return residual


def residual_prediction(
    state,
    history,
    *,
    baseline,
    method,
    workers,
    beam_current_A,
    target_time,
    machine_state=None,
    policy=Weighting(),
):
    # Filter and sort first, then select the newest one, newest three, or all
    # eligible records. 
    records = eligible_history(history, target_time=target_time, require_ids=True)
    count = {"last": 1, "mean3": 3}.get(method)
    if method not in {"last", "mean3", "ewma"}:
        raise ValueError(f"unknown history estimator {method!r}")
    if not records or (count is not None and len(records) < count):
        raise RuntimeError("insufficient earlier ID-associated LOCO measurements")
    selected = records[-count:] if count is not None else records
    weights = (
        history_weights(selected, state, beam_current_A, target_time, policy)
        if method == "ewma"
        else np.ones(len(selected)) / len(selected)
    )
    residuals = [
        _cached_residual(baseline, item, workers=workers) for item in selected
    ]
    # Calculate the physics once at the requested state, then add the weighted
    # historical residual
    current = baseline(
        state,
        [],
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
        machine_state=machine_state,
    )
    return add_orm(current, weighted_orm(residuals, weights))
