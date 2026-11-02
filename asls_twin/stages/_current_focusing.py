"""Uniform vertical current focusing, represented by small kicks around the ring."""

from functools import lru_cache

import numpy as np

from ..calculations import family_indices, load_ring

# Open-chamber LOCO/DC slope from Tan and Dowd, IPAC2018 TUPMK010.
CURRENT_WAKE_TUNE_SLOPE_PER_A = -0.0359
CURRENT_WAKE_ANCHOR_A = 0.200
CURRENT_WAKE_MAX_DRIFT_M = 0.25


def _current_delta(beam_current_A):
    if (
        beam_current_A is None
        or not np.isfinite(beam_current_A)
        or not 0 < beam_current_A <= 1
    ):
        raise ValueError("Stage 2h requires beam_current_A in (0, 1]")
    return float(beam_current_A) - CURRENT_WAKE_ANCHOR_A


def _current_mesh(ring, max_drift_m=CURRENT_WAKE_MAX_DRIFT_M):
    """Symmetric length-weighted kicks at each finite segment's faces.

    Long drifts are divided; magnetic elements keep their existing fine
    segmentation, fringe fields, radiation and family identities. The new
    lenses have zero geometric length. No device or corrector is duplicated.
    """
    import at

    if not np.isfinite(max_drift_m) or max_drift_m < 0.025:
        raise ValueError("current mesh drift step must be at least 0.025 m")
    if any(e.FamName == "CurrentWake" for e in ring):
        raise ValueError("whole-ring current contribution is already applied")
    elements = []
    for element in ring:
        length = float(element.Length)
        if length < 0 or not np.isfinite(length):
            raise ValueError("current mesh requires finite nonnegative lengths")
        count = (int(np.ceil(length / max_drift_m))
                 if element.PassMethod == "DriftPass" and length else 1)
        if count > 1000 or len(elements) + 3 * count > 20000:
            raise ValueError("current mesh exceeds the bounded lattice size")
        parts = element.divide(np.full(count, 1.0 / count)) if count > 1 else (element,)
        for part in parts:
            if part.Length == 0:
                elements.append(part)
                continue
            # Half the focusing before the segment, half after it.
            entrance = at.M66("CurrentWake", np.eye(6))
            exit_lens = at.M66("CurrentWake", np.eye(6))
            entrance.CurrentWakeLength = exit_lens.CurrentWakeLength = float(part.Length) / 2
            elements.extend((entrance, part, exit_lens))
    return at.Lattice(lambda _: iter(elements), iterator=ring.attrs_filter)


@lru_cache(maxsize=1)
def whole_ring_current_calibration():
    """Fix the uniform kick density from the nominal lattice, never LOCO fits."""
    ring = _current_mesh(load_ring())
    refs = np.asarray([i for i,e in enumerate(ring) if e.FamName == "CurrentWake"], dtype=np.uint32)
    weights = np.asarray([ring[int(i)].CurrentWakeLength for i in refs])
    beta = np.asarray(ring.get_optics(refpts=refs)[2].beta)[:, 1]
    beta_integral = float(beta @ weights)
    if not np.isfinite(beta_integral) or beta_integral <= 0:
        raise RuntimeError("invalid whole-ring beta integral")
    coefficient = -4 * np.pi * CURRENT_WAKE_TUNE_SLOPE_PER_A / beta_integral
    return coefficient, beta_integral


def apply_whole_ring_current(ring, beam_current_A):
    """Add the same vertical kick density per metre around the entire ring."""
    delta = _current_delta(beam_current_A)
    if any(e.FamName == "CurrentWake" for e in ring):
        raise ValueError("whole-ring current contribution is already applied")
    if delta == 0:
        return ring, family_indices(ring)
    coefficient, _ = whole_ring_current_calibration()
    ring = _current_mesh(ring)
    for element in ring:
        if element.FamName == "CurrentWake":
            element.M66[3, 2] = coefficient * delta * element.CurrentWakeLength
    return ring, family_indices(ring)
