"""System-selectable forward ORM generation, with no machine I/O.

SOFB delegates to the existing implementation to retain its frozen baseline
fingerprints. FOFB defaults to the periodic-orbit derivative; the bipolar
whole-ring kernel remains available for numerical verification.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from . import calculations, config
from .systems import get_system
from .types import ORM


def corrector_indices(indices, *, system="SOFB"):
    spec = get_system(system)
    def select(selection):
        return np.sort(np.concatenate([
            calculations._family(indices, family)[offset::stride]
            for family, stride, offset in selection]))
    horizontal, vertical = select(spec.horizontal_selection), select(spec.vertical_selection)
    if (horizontal.shape != (spec.n_horizontal,) or vertical.shape != (spec.n_vertical,)):
        raise ValueError(f"{system} lattice corrector counts differ from the device contract")
    return horizontal, vertical


def _linear_blocks(ring, bpm, horizontal, vertical):
    """Differentiate the 6D periodic orbit, including each thick kick element.

    One closed orbit and one set of transfer matrices replace a pair of
    whole-ring orbit solves per corrector. Local element differentiation uses
    a 1e-7 rad step. This is the infinitesimal ORM, not the finite 1e-5 rad
    bipolar approximation; both methods retain the lattice's radiation/RF.
    Adapted from analysis/asls_stage8_calibrate.py:linear_response.
    """
    import at
    from at.tracking.utils import variable_refs

    if np.any(variable_refs(ring)):
        raise ValueError("linear ORM requires a lattice without variable elements")
    refs = np.arange(len(ring) + 1, dtype=np.uint32)
    locations = np.concatenate((horizontal, vertical))
    sources = np.empty((len(locations), 6))
    with calculations._hoisted_tracking(ring):
        orbit0, orbits = at.find_orbit6(ring, refpts=refs)
        turn, transfers = at.find_m66(ring, refpts=refs, orbit=orbit0)
        for column, index in enumerate(locations):
            element = ring[int(index)]
            saved = np.array(element.KickAngle, float)
            plane = 0 if column < len(horizontal) else 1
            incoming = np.array(orbits[int(index), :, None], order="F", copy=True)
            try:
                element.KickAngle[plane] = saved[plane] + 1e-7
                plus = at.element_track(element, incoming, in_place=False).ravel()
                element.KickAngle[plane] = saved[plane] - 1e-7
                minus = at.element_track(element, incoming, in_place=False).ravel()
            finally:
                element.KickAngle[:] = saved
            sources[column] = (plus - minus) / 2e-7
    # Map each local kick derivative back to the origin. The closed-orbit
    # term solves (I - M) dx = M s; downstream BPMs also receive the direct s.
    initial = np.linalg.solve(transfers[locations + 1], sources[..., None])[..., 0].T
    periodic = np.linalg.solve(np.eye(6) - turn, turn @ initial)
    bpm_transfer = transfers[bpm][:, [0, 2], :]
    values = np.einsum("bij,jk->bik", bpm_transfer, periodic)
    values += (np.einsum("bij,jk->bik", bpm_transfer, initial)
               * (bpm[:, None] > locations[None, :])[:, None, :])
    split = len(horizontal)
    return dict(Rxx=values[:, 0, :split], Ryx=values[:, 1, :split],
                Rxy=values[:, 0, split:], Ryy=values[:, 1, split:])


def build_orm(ring, *, system="SOFB", workers=1, indices=None, method=None):
    """Build a fresh matrix: FOFB defaults to linear, SOFB retains bipolar."""
    spec = get_system(system)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    method = method or ("linear" if system == "FOFB" else "bipolar")
    if method not in ("linear", "bipolar"):
        raise ValueError("ORM method must be linear or bipolar")
    if system == "SOFB" and method == "bipolar":
        return calculations.build_orm(ring, workers=workers, indices=indices)
    # Keep the initial FOFB path bounded and deterministic on facility desktops.
    if workers != 1:
        raise ValueError("this ORM builder requires workers=1")
    indices = calculations.family_indices(ring) if indices is None else indices
    bpm = calculations._family(indices, "BPM")
    if bpm.shape != (spec.n_bpm,):
        raise ValueError(f"{system} lattice BPM count differs from the device contract")
    horizontal, vertical = corrector_indices(indices, system=system)
    if method == "linear":
        result = _linear_blocks(ring, bpm, horizontal, vertical)
        return ORM(*(result[name] for name in ("Rxx", "Ryx", "Rxy", "Ryy")),
                   "m/rad", system).check()
    result = {name: np.empty(shape) for name, shape in spec.shapes.items()}
    with calculations._hoisted_tracking(ring):
        for plane, locations, names in ((0, horizontal, ("Rxx", "Ryx")),
                                        (1, vertical, ("Rxy", "Ryy"))):
            for column, index in enumerate(locations):
                x, y = calculations._column(ring, int(index), plane, bpm)
                result[names[0]][:, column], result[names[1]][:, column] = x, y
    return ORM(*(result[name] for name in ("Rxx", "Ryx", "Rxy", "Ryy")),
               "m/rad", system).check()


@lru_cache(maxsize=6)
def _fofb_static(mode, method):
    orm = build_orm(calculations.load_ring(mode), system="FOFB",
                    indices=calculations._cached_family_indices(mode), method=method)
    for name in ("Rxx", "Ryx", "Rxy", "Ryy"):
        getattr(orm, name).flags.writeable = False
    return orm


def static_at_orm(*, system="SOFB", mode=config.DEFAULT_MACHINE_MODE, workers=1, method=None):
    get_system(system)
    if system == "SOFB":
        if method not in (None, "bipolar"):
            return build_orm(calculations.load_ring(mode), system=system,
                             workers=workers, method=method)
        return calculations.static_at_orm(mode=mode, workers=workers)
    if workers != 1:
        raise ValueError("FOFB generation currently requires workers=1")
    return _fofb_static(mode, method or "linear").copy()


def id_aware_at_orm(state, *, system="SOFB", mode=config.DEFAULT_MACHINE_MODE, workers=1, method=None):
    get_system(system)
    if system == "SOFB" and method in (None, "bipolar"):
        return calculations.id_aware_at_orm(state, mode=mode, workers=workers)
    return build_orm(calculations.apply_id_state(state, mode=mode), system=system,
                     workers=workers, indices=calculations._cached_family_indices(mode), method=method)


def at_to_ofb(orm, *, scales=None):
    """Convert to nm/A only with explicit FOFB per-plane calibration factors."""
    orm.check()
    if orm.system == "SOFB":
        if scales is not None:
            raise ValueError("SOFB uses its existing configured calibration")
        return calculations.at_to_ofb(orm)
    if orm.units == "nm/A":
        return orm
    if orm.units != "m/rad":
        raise ValueError(f"unsupported ORM units {orm.units!r}")
    factors = np.asarray(scales, float)
    if factors.shape != (2,) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("FOFB nm/A output requires explicit positive finite scales=(horizontal, vertical)")
    return ORM(orm.Rxx * factors[0], orm.Ryx * factors[0],
               orm.Rxy * factors[1], orm.Ryy * factors[1], "nm/A", "FOFB").check()
