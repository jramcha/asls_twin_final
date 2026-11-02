"""Stage 6h: Stage 3h-weighted plus the frozen residual ridge."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .. import config
from ..calculations import regression_features
from ._history import Weighting
from .stage3h_weighted import calculate as calculate_stage3h_weighted


ARTIFACT = Path(config.DATA) / "stage6h_mean3_ridge.npz"
ARTIFACT_SHA256 = "cf496416213a3803921df487409048472c1bbe513fc4554148ee36081e4b26e5"
GAP_SCALING = "physical_b_squared_range_v1"


def gap_feature_scaling():
    """Fixed B-squared lower bounds and spans for configured physical gaps.

    Open gap maps to zero and closed gap to one. No training observations
    enter these scales. Field-controlled IDs retain their fitted scaling.
    """
    indices, lower, spans = [], [], []
    for j, name in enumerate(config.ID_NAMES):
        if name not in config.GAP_FEATURE_PERIOD_MM:
            continue
        bounds = np.asarray(config.ID_READBACK_RANGES[name], dtype=float)
        values = np.exp(-2 * np.pi * bounds / config.GAP_FEATURE_PERIOD_MM[name])
        span = float(values[0] - values[1])
        if not np.isfinite(values).all() or span <= 0:
            raise ValueError("invalid physical gap feature bounds: " + name)
        indices.append(j)
        lower.append(values[1])
        spans.append(span)
    return np.asarray(indices, dtype=int), np.asarray(lower), np.asarray(spans)

def calculate(
    state,
    history,
    *,
    workers=1,
    beam_current_A=None,
    target_time=None,
    machine_state=None,
    **context,
):
    del context
    if machine_state is not None:
        raise ValueError("Stage 6h is bound to the nominal Stage 2h calibration")
    if hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() != ARTIFACT_SHA256:
        raise RuntimeError("Stage 6h artifact SHA-256 mismatch")
    with np.load(ARTIFACT, allow_pickle=False) as artifact:
        if (
            artifact["centre"].shape != (len(config.ID_NAMES),)
            or artifact["scale"].shape != (len(config.ID_NAMES),)
            or artifact["coef"].shape
            != (len(config.ID_NAMES), config.N_BPM * (config.N_HCM + config.N_VCM))
            or artifact["intercept"].shape
            != (config.N_BPM * (config.N_HCM + config.N_VCM),)
        ):
            raise RuntimeError("invalid Stage 6h artifact dimensions")
        if not all(
            np.isfinite(artifact[name]).all()
            for name in ("centre", "scale", "coef", "intercept")
        ) or np.any(artifact["scale"] <= 0):
            raise RuntimeError("invalid Stage 6h artifact values")
        indices, lower, spans = gap_feature_scaling()
        if ("gap_scaling" not in artifact
                or str(artifact["gap_scaling"]) != GAP_SCALING
                or not np.allclose(artifact["centre"][indices], lower, rtol=1e-13, atol=1e-15)
                or not np.allclose(artifact["scale"][indices], spans, rtol=1e-13, atol=1e-15)):
            raise RuntimeError("Stage 6h artifact requires physical gap scaling")
        correction = artifact["intercept"] + (
            (regression_features(state) - artifact["centre"]) / artifact["scale"]
        ) @ artifact["coef"]
    base = calculate_stage3h_weighted(
        state,
        history,
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
        # The frozen ridge was fitted against the original history policy.
        policy=Weighting(),
    )
    split = config.N_BPM * config.N_HCM
    result = base.copy()
    result.Rxx[:] += correction[:split].reshape(result.Rxx.shape)
    result.Ryy[:] += correction[split:].reshape(result.Ryy.shape)
    return result.check()
