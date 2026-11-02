"""Stage 3h-weighted: Stage 2h plus ID/current/time-weighted signed residuals."""

import logging
from math import isfinite

from .. import config
from ._history import Weighting, eligible_history, residual_prediction
from .stage2h import calculate as calculate_stage2h


def _valid_current(value):
    try:
        return isfinite(value) and 0 < value <= 1
    except (TypeError, ValueError):
        return False


def history_policy():
    """Prefer nearby ID states without oversensitivity to stationary IDs."""
    return Weighting(
        half_life_days=config.STAGE3H_HISTORY_HALF_LIFE_DAYS,
        id_bandwidth=config.STAGE3H_HISTORY_ID_BANDWIDTH,
        current_bandwidth=config.STAGE3H_HISTORY_CURRENT_BANDWIDTH,
        span_floor_fraction=config.STAGE3H_HISTORY_SPAN_FLOOR_FRACTION,
        span_ceiling_fraction=config.STAGE3H_HISTORY_SPAN_CEILING_FRACTION,
        feature_scale_floor=config.STAGE3H_HISTORY_SCALE_FLOOR,
        current_scale_floor_A=config.STAGE3H_HISTORY_CURRENT_SCALE_FLOOR_A,
        sum_id_distances=True,
    ).check()


def calculate(
    state,
    history,
    *,
    workers=1,
    beam_current_A=None,
    target_time=None,
    machine_state=None,
    policy=None,
    **context,
):
    del context
    if not _valid_current(beam_current_A):
        raise ValueError("Stage 3h-weighted requires beam_current_A in (0, 1]")
    records = eligible_history(history, target_time=target_time, require_ids=True)
    usable = [item for item in records if _valid_current(item.beam_current_A)]
    excluded = [item for item in records if not _valid_current(item.beam_current_A)]
    if excluded:
        logging.getLogger(__name__).warning(
            "Stage 3h-weighted excluded %d historical LOCO records with missing "
            "or invalid beam current: %s",
            len(excluded),
            ", ".join(str(item.path) for item in excluded),
        )
    if not usable:
        raise RuntimeError(
            "Stage 3h-weighted requires at least one earlier ID-associated "
            "LOCO measurement with valid beam current"
        )
    return residual_prediction(
        state,
        usable,
        baseline=calculate_stage2h,
        method="ewma",
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
        machine_state=machine_state,
        policy=history_policy() if policy is None else policy,
    )
