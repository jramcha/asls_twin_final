# Last reviewed: 20/09/2026
"""The four selectable update policies."""

from __future__ import annotations #this is allowing the python file to import a feature from a newer language (future language)
# annotations are handled slightly more optimally (lazily, not evaluating them unless required).

import numpy as np #normal, np

from .config import UpdateConfig
from . import config
from .safety import normalized_id_strengths, relative_orm_change_percent
from .types import ORM, UpdateTracker

#frobenius distance between last refreshed and current ID state as a PV from epics. note for jai check for making this generalised for ID count from config. 
def id_frob_percent(current: np.ndarray, baseline: np.ndarray) -> float:
    current = np.asarray(current, float)
    baseline = np.asarray(baseline, float)
    if current.shape != (len(config.ID_NAMES),) or baseline.shape != (len(config.ID_NAMES),):
        raise ValueError(f"normalized ID strength vectors must both have shape {(len(config.ID_NAMES),)}")
    return float(100.0 * np.linalg.norm(current - baseline) / np.sqrt(len(config.ID_NAMES)))


def id_frob_cubed_percent(current: np.ndarray, baseline: np.ndarray) -> float:
    current = np.asarray(current, float)
    baseline = np.asarray(baseline, float)
    if current.shape != (len(config.ID_NAMES),) or baseline.shape != (len(config.ID_NAMES),):
        raise ValueError(f"normalized ID strength vectors must both have shape {(len(config.ID_NAMES),)}")
    return float(100.0 * np.sum(np.abs(current**3 - baseline**3)) / len(config.ID_NAMES))


# Review synthesis: choose one configured trigger, not an OR of all modes.
# ID distances use normalized magnetic strengths against the last accepted
# baseline. The separate push-attempt rate cap is enforced in main.run.
def should_update(
    update: UpdateConfig,
    tracker: UpdateTracker,
    state: dict[str, float],
    *,
    now_monotonic: float,
    candidate: ORM | None = None,
    reference: ORM | None = None,
) -> tuple[bool, float]:
    threshold = update.selected_threshold
    if update.mode == "time":
        value = (
            float("inf")
            if tracker.last_update_monotonic is None
            else now_monotonic - tracker.last_update_monotonic
        )
    elif update.mode in {"idFrob", "idFrobCubed"}:
        strengths = normalized_id_strengths(state)
        if tracker.accepted_id_strengths is None:
            value = float("inf")
        elif update.mode == "idFrob":
            value = id_frob_percent(strengths, tracker.accepted_id_strengths)
        else:
            value = id_frob_cubed_percent(strengths, tracker.accepted_id_strengths)
    else:
        if update.mode != "ormPredFrob":
            raise ValueError(f"unsupported update mode {update.mode!r}")
        if candidate is None or reference is None:
            return False, 0.0
        value = relative_orm_change_percent(candidate, reference)
    return value >= threshold, value


# A trigger check alone must not reset the baseline. The service advances
# it after a verified write or an accepted dry run, not after a failed write.
def advance_baseline(
    tracker: UpdateTracker,
    state: dict[str, float],
    candidate: ORM,
    *,
    now_monotonic: float,
) -> None:
    """Advance only after the caller has accepted the candidate."""
    tracker.last_update_monotonic = now_monotonic
    tracker.accepted_id_strengths = normalized_id_strengths(state)
    tracker.accepted_orm = candidate
