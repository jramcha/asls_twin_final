"""ID magnetic focusing, feed-forward and the whole-ring current effect."""

from functools import lru_cache
import hashlib

import numpy as np
import scipy.io as sio

from .. import config
from ..calculations import _interp_clamped, family_indices, load_ring
from ..safety import validate_id_state
from ._current_focusing import apply_whole_ring_current, _current_delta


M66_FILE = config.M66_FITS_MAT
QUAD_FF_FILE = config.QUAD_FF_MAT
M66_SHA256 = "b92f11b906dd7762137694e546ee3583049867a2e565921f1780121edfe73be7"
QUAD_FF_SHA256 = "527d1f9a874154de5f5baa419406bcef6c699e3fd8be157e323743ef2c8f48bb"
M66_KEY = {
    "ID03": "SR03ID01",
    "ID04": "SR04ID01",
    "ID05": "SR05ID01",
    "ID08": "SR08ID01",
    "ID12": "SR12ID01",
    "ID13": "SR13ID01",
}
QUAD_FF_KEY = {
    "ID02": "SR02ID01",
    "ID03": "SR03ID01",
    "ID05": "SR05ID01",
    "ID08": "SR08ID01",
    "ID12": "SR12ID01",
    "ID13": "SR13ID01",
}
M66_KY_OPEN_REBASE = {
    "ID03": -0.001337366494860519,
    "ID04": -0.004721097779144359,
    "ID05": -0.0005539004395746923,
    "ID12": -0.002821798997499191,
    "ID13": -0.0016887278091538254,
}


@lru_cache(maxsize=1)
def _tables():
    if hashlib.sha256(M66_FILE.read_bytes()).hexdigest() != M66_SHA256:
        raise RuntimeError("Stage 2h M66 table SHA-256 mismatch")
    if hashlib.sha256(QUAD_FF_FILE.read_bytes()).hexdigest() != QUAD_FF_SHA256:
        raise RuntimeError("Stage 2h quadrupole feed-forward SHA-256 mismatch")
    return (
        sio.loadmat(M66_FILE, squeeze_me=True, struct_as_record=False),
        sio.loadmat(QUAD_FF_FILE, squeeze_me=True, struct_as_record=False),
    )


def _parameters(state):
    state = validate_id_state(state)
    m66, feed_forward = _tables()
    parameters = {
        "Kx": np.zeros(len(config.ID_NAMES)),
        "Ky": np.zeros(len(config.ID_NAMES)),
        "dKqfa": np.zeros(config.FAMILY_COUNTS["QFA"]),
        "dKqda": np.zeros(config.FAMILY_COUNTS["QDA"]),
        "dKqfb": np.zeros(config.FAMILY_COUNTS["QFB"]),
    }
    for position, name in enumerate(config.ID_NAMES):
        if name in M66_KEY:
            fit = m66[M66_KEY[name]]
            for plane in ("Kx", "Ky"):
                parameters[plane][position] = _interp_clamped(
                    state[name], np.ravel(fit.Gap), np.ravel(getattr(fit, plane))
                )
        if name in QUAD_FF_KEY:
            table = feed_forward[QUAD_FF_KEY[name]]
            for key, field in (
                ("dKqfa", "kqfa"),
                ("dKqda", "kqda"),
                ("dKqfb", "kqfb"),
            ):
                rows = np.asarray(getattr(table, field), float)
                if rows.shape != (len(parameters[key]), len(np.ravel(table.gaps))):
                    raise RuntimeError("Stage 2h feed-forward table shape mismatch")
                for row in range(len(parameters[key])):
                    parameters[key][row] += _interp_clamped(
                        state[name], np.ravel(table.gaps), rows[row]
                    )
    return parameters


def build_stage2h_ring(state, beam_current_A):
    """Build ID optics, then add current focusing throughout the ring."""
    _current_delta(beam_current_A)
    ring, _ = build_id_ring(state)
    return apply_whole_ring_current(ring, beam_current_A)


def build_stage2ha_ring(state, beam_current_A):
    """Shared baseline for callers using the former Stage 2ha entry point."""
    return build_stage2h_ring(state, beam_current_A)


def build_id_ring(state):
    """Build the ID and feed-forward optics without the current correction."""
    ring = load_ring()
    indices = family_indices(ring)
    parameters = _parameters(state)
    markers = indices["ID"]
    if len(markers) != config.N_ID_MARKERS:
        raise RuntimeError("Stage 2h found an unexpected ID marker count")
    for name, marker_index in config.ID_MARKER_INDEX.items():
        position = config.ID_NAMES.index(name)
        matrix = np.eye(6)
        matrix[1, 0] = parameters["Kx"][position]
        matrix[3, 2] = (
            parameters["Ky"][position]
            + M66_KY_OPEN_REBASE.get(name, 0.0)
        )
        ring[int(markers[marker_index])].M66 = matrix
    for family, key in (
        ("QFA", "dKqfa"),
        ("QDA", "dKqda"),
        ("QFB", "dKqfb"),
    ):
        if len(indices[family]) != config.FAMILY_COUNTS[family]:
            raise RuntimeError(f"Stage 2h found an unexpected {family} count")
        for row, index in enumerate(indices[family]):
            ring[int(index)].PolynomB[1] = (
                config.MODE_SETTINGS[config.DEFAULT_MACHINE_MODE][family]
                + parameters[key][row]
            )
    return ring, indices
