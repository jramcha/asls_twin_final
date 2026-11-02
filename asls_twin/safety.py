# Last reviewed: 20/09/2026
"""Strict input, matrix, distance, and reference safety gates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from . import config
from .types import ORM, SafetyDecision

# Put every input into the configured ID order and check the full set of
# readbacks. 
def validate_id_state(state: Mapping[str, float] | np.ndarray) -> dict[str, float]:
    if not isinstance(state, Mapping):
        values = np.asarray(state, float).ravel()
        if values.shape != (len(config.ID_NAMES),):
            raise ValueError(f"expected {len(config.ID_NAMES)} ID values, got shape {values.shape}")
        state = dict(zip(config.ID_NAMES, values))
    missing = [name for name in config.ID_NAMES if name not in state]
    extra = sorted(set(state).difference(config.ID_NAMES))
    if missing or extra:
        raise ValueError(f"ID state keys differ: missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for name in config.ID_NAMES:
        value = float(state[name])
        low, high = config.ID_READBACK_RANGES[name]
        if not np.isfinite(value):
            raise ValueError(f"{name} readback is not finite")
        if not low <= value <= high:
            raise ValueError(f"{name} readback {value:g} is outside [{low:g}, {high:g}]")
        result[name] = value
    return result

# Load each ID conversion table once. 
@lru_cache(maxsize=None)
def _gap_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = config.GAP_FIELD_TABLES.get(name)
    if path is None or not Path(path).exists():
        raise RuntimeError(f"missing gap-to-field conversion data for {name}")
    table = np.loadtxt(path, delimiter="\t", skiprows=1)
    if table.ndim != 2 or table.shape[1] < 3 or not np.isfinite(table[:, :3]).all():
        raise ValueError(f"malformed gap-to-field table for {name}")
    gap = np.asarray(table[:, 0], float)
    field = np.hypot(table[:, 1], table[:, 2])
    if name in config.GAP_FIELD_ENDPOINTS:
        endpoint_gap, endpoint_field = config.GAP_FIELD_ENDPOINTS[name]
        if endpoint_gap < gap.min() or endpoint_gap > gap.max():
            gap = np.append(gap, endpoint_gap)
            field = np.append(field, endpoint_field)
    order = np.argsort(gap)
    gap, field = gap[order], field[order]
    if np.any(np.diff(gap) <= 0) or field.max() <= 0:
        raise ValueError(f"invalid gap-to-field table for {name}")
    return gap, field

# These normalized strengths are used by the ID update triggers.
def normalized_id_strengths(state: Mapping[str, float] | np.ndarray) -> np.ndarray:
    """Convert all eight readbacks to normalized magnetic strength."""
    strict = validate_id_state(state)
    result = np.empty(len(config.ID_NAMES))
    for j, name in enumerate(config.ID_NAMES):
        value = strict[name]
        kind = config.ID_PVS[name][1]
        # Divide direct field readbacks by their confirmed maximum. For gap
        # readbacks, interpolate the field first and normalize by the table maximum.
        if kind == "field":
            maximum = config.FIELD_MAX_T.get(name)
            if maximum is None or maximum <= 0:
                raise RuntimeError(f"missing confirmed maximum field for {name}")
            result[j] = value / maximum
        elif kind == "gap":
            gap, field = _gap_table(name)
            if value < gap.min() or value > gap.max():
                raise ValueError(
                    f"{name} gap {value:g} is outside conversion support "
                    f"[{gap.min():g}, {gap.max():g}]"
                )
            result[j] = np.interp(value, gap, field) / field.max()
        else:
            raise RuntimeError(f"unknown ID readback kind {kind!r} for {name}")
    if not np.isfinite(result).all():
        raise ValueError("normalized ID strengths contain non-finite values")
    return result

# Require finite, correctly shaped response blocks in deployment units.
def validate_candidate(candidate: ORM) -> ORM:
    candidate.check(finite=True)
    if candidate.units != "nm/A":
        raise ValueError(f"candidate units must be exactly 'nm/A', got {candidate.units!r}")
    return candidate


# Review synthesis: compare the flattened Rxx/Ryy blocks that SOFB consumes.
# This is a relative matrix-change gate, not a beam-stability measurement.
def relative_orm_change_percent(candidate: ORM, reference: ORM) -> float:
    validate_candidate(candidate)
    validate_candidate(reference)
    denominator = np.linalg.norm(reference.upload_vector)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("reference ORM has zero or invalid Frobenius norm")
    return float(
        100.0 * np.linalg.norm(candidate.upload_vector - reference.upload_vector) / denominator
    )


# A rejected candidate keeps the reference available but sets should_write
# false. Acceptance here does not bypass the two EPICS write arms.
def choose_safe_orm(
    candidate: ORM,
    reference: ORM | None,
    *,
    max_candidate_change_percent: float = 3.0,
) -> SafetyDecision:
    """Accept a candidate or fall back to the in-use/last verified reference."""
    if not np.isfinite(max_candidate_change_percent) or max_candidate_change_percent <= 0:
        raise ValueError("max_candidate_change_percent must be positive and finite")
    try:
        validate_candidate(candidate)
    except (TypeError, ValueError) as exc:
        return SafetyDecision(reference, False, False, f"invalid candidate: {exc}")
    if reference is None:
        return SafetyDecision(None, False, False, "no in-use or verified reference ORM")
    try:
        change = relative_orm_change_percent(candidate, reference)
    except (TypeError, ValueError) as exc:
        return SafetyDecision(reference, False, False, f"invalid reference: {exc}")
    # Keep the reference when the proposed step is too large.
    if change > max_candidate_change_percent:
        return SafetyDecision(
            reference,
            False,
            False,
            f"candidate change {change:.6g}% exceeds {max_candidate_change_percent:g}%",
            change,
        )
    # A whole-matrix norm can hide the loss of one BPM or corrector channel.
    # Compare each response vector against the channel it will replace.
    for block in ("Rxx", "Ryy"):
        old = getattr(reference, block)
        new = getattr(candidate, block)
        for axis, channel in ((0, "corrector"), (1, "BPM")):
            old_norm = np.linalg.norm(old, axis=axis)
            delta_norm = np.linalg.norm(new - old, axis=axis)
            changed = (old_norm > 0) & (delta_norm >= old_norm * 0.5)
            if np.any(changed):
                index = int(np.flatnonzero(changed)[0]) + 1
                return SafetyDecision(reference, False, False,
                    f"{block} {channel} {index} response changed by at least 50%", change)
    return SafetyDecision(candidate, True, True, "candidate accepted", change)
