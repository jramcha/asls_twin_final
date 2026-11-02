# Last reviewed: 19/09/2026
"""Strict EPICS reads and transactional SOFB matrix writes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .safety import validate_candidate, validate_id_state
from .types import ORM, SafetyDecision, WriteResult


DEFAULT_TIMEOUT = config.DEFAULT_EPICS_TIMEOUT


class UnsafeMachineStateError(RuntimeError):
    """A matrix write or its rollback cannot be trusted; stop until rearmed."""

# Sam note on PV objects: 
# An Epics Process Variable is implemented as a python PV object, which 
# provides a way to interact with EPICS through  channel access, 
# i.e get or set PV objects (such as BPM readings or RM values)


# Imports EPICS caget function which is used to retrieve and returns the value of named PV.
# epics.caget(pvname[, as_string=False[, count=None[, as_numpy=True[, timeout=None[, use_monitor=True]]]]])
#Parameters:
    # pvname – name of Epics Process Variable.
    # as_string (True/False) – whether to return string representation of the PV value.
    # count (integer or None) – number of elements to return for array data.
    # as_numpy (True/False) – whether to return the Numerical Python representation for array data.
    # timeout (float or None) – maximum time to wait (in seconds) for value before returning None.
    # use_monitor (True/False) – whether to rely on monitor callbacks or explicitly get value now.
def _default_caget() -> Callable:
    try:
        from epics import caget
    except ImportError as exc:
        raise RuntimeError("pyepics is required for live EPICS reads") from exc
    # A monitor event may arrive after caput(wait=True) completes. Safety
    # verification must read the server, never a possibly stale monitor cache.
    from functools import partial
    return partial(caget, use_monitor=False)


# Imports EPICS caget function which is used to set the values of named PV objects
# epics.caput(pvname, value[, wait=False[, timeout=60]])
# Parameters:
# pvname – name of Epics Process Variable
# value – value to send.
# wait (True/False) – whether to wait until the processing has completed.
# timeout (double) – how long to wait (in seconds) for put to complete before giving up.
# Return type = integer
def _default_caput() -> Callable:
    try:
        from epics import caput
    except ImportError as exc:
        raise RuntimeError("pyepics is required for live EPICS writes") from exc
    return caput


# Reads the current values of all configured ID PVs and returns them
def read_id_state(
    *, timeout: float = DEFAULT_TIMEOUT, caget: Callable | None = None
) -> dict[str, float]:
    """Read all eight IDs strictly; no resting-value substitution exists."""
    get = caget or _default_caget()

    # Stores successfuly read ID values in the state dictionary, and any missing PVs in the missing list
    state: dict[str, float] = {}
    missing: list[str] = []

    # For each ID in the ID_NAMES list, read the corresponding ID PV
    for name in config.ID_NAMES:
        pv = config.ID_PVS[name][0]
        value = get(pv, timeout=timeout)
        if value is None:
            missing.append(f"{name} ({pv})")
        else:
            array = np.asarray(value)
            if array.size != 1:
                raise RuntimeError(f"{pv} returned {array.size} values, expected one")
            state[name] = float(array.ravel()[0])
    if missing:
        raise RuntimeError(f"ID PVs did not answer: {', '.join(missing)}")
    return validate_id_state(state)


# Similar to read_id_state, but reads the beam current PV 
def read_beam_current(
    *, timeout: float = DEFAULT_TIMEOUT, caget: Callable | None = None
) -> float:
    value = (caget or _default_caget())(config.BEAM_CURRENT_PV, timeout=timeout)
    if value is None or np.asarray(value).size != 1 or not np.isfinite(float(value)):
        raise RuntimeError("beam-current PV did not return one finite value")
    return float(value)


# Helper function used to read a PV that returens an array, also checks if data is valid before returning np array
def _read_vector(pv: str, size: int, *, timeout: float, caget: Callable) -> np.ndarray:
    value = caget(pv, timeout=timeout)
    if value is None:
        raise RuntimeError(f"{pv} did not answer")
    vector = np.asarray(value, float).ravel()
    if vector.shape != (size,):
        raise RuntimeError(f"{pv} returned {vector.size} values, expected {size}")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"{pv} returned non-finite values")
    return vector

# reads the  horizontal and vertical ORMS, converts the PV arrays into 2-D NumPy matrices and store both horizontal and vertical data in an ORM object
def read_current_orm(
    *, timeout: float = DEFAULT_TIMEOUT, caget: Callable | None = None
) -> ORM:
    get = caget or _default_caget()

    # Uses _read_vector function to read the horizontal ORM
    horizontal = _read_vector(
        config.OFB_H_MATRIX_PV, #retreives the horizontal ORM PV name from the config file
        config.N_BPM * config.N_HCM, #determines required size of array by multiplying the number of BPMs and HCMs
        timeout=timeout,
        caget=get,
    )
    # Uses _read_vector function to read the vertical ORM
    vertical = _read_vector(
        config.OFB_V_MATRIX_PV, #retreives the vertical ORM PV name from the config file
        config.N_BPM * config.N_VCM, #determines required size of array by multiplying the number of BPMs and HCMs
        timeout=timeout,
        caget=get,
    )
    return ORM(
        horizontal.reshape(config.N_BPM, config.N_HCM), #reshapes the horizontal ORM into a matrix
        np.zeros((config.N_BPM, config.N_HCM)),
        np.zeros((config.N_BPM, config.N_VCM)),
        vertical.reshape(config.N_BPM, config.N_VCM), #reshapes the vertical ORM into a matrix
        "nm/A",
    ).check()

# takes an ORM object, validates it, and then seperates and returns the two horizontal and vertical ORMs
def _vectors(orm: ORM) -> tuple[np.ndarray, np.ndarray]:
    if orm.system != "SOFB":
        raise ValueError("the EPICS writer accepts SOFB matrices only")
    validate_candidate(orm)
    return orm.Rxx.ravel(order="C"), orm.Ryy.ravel(order="C")

# helper that writes a NumPy array to a PV using caput(), waits for the write operation to complete
def _put(caput: Callable, pv: str, value: np.ndarray, timeout: float) -> None:
    result = caput(pv, value, wait=True, timeout=timeout)
    if result != 1:
        raise RuntimeError(f"write to {pv} did not complete")
[]

# checks that the ORM currently stored in EPICS matches the ORM that the code expected to write. 
# does this by reading the matrix from epics and comparing to expected
def _verify(
    expected: ORM,
    *,
    caget: Callable,
    timeout: float,
    rtol: float,
    atol: float,
) -> None:
    actual = read_current_orm(timeout=timeout, caget=caget) # Gets current ORM from EPICS
    if not np.allclose(actual.Rxx, expected.Rxx, rtol=rtol, atol=atol): #allclose checks that the ellements of the matrices are close wihin a tolerance
        raise RuntimeError("horizontal response-matrix readback did not verify")
    if not np.allclose(actual.Ryy, expected.Ryy, rtol=rtol, atol=atol):
        raise RuntimeError("vertical response-matrix readback did not verify")

# Main function that writes the ORM to EPICS, and restores the previous ORM if the write fails.
# Review synthesis (Sam): horizontal and vertical PV writes are sequential,
# not an atomic hardware transaction. Readback verifies the pair; a failure
# attempts to restore and verify both previous vectors immediately.
def write_orm(
    value: ORM | SafetyDecision,
    *,
    writes_enabled: bool = False,
    enable_writes: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    readback_rtol: float = 1e-9,
    readback_atol_nm_per_a: float = 1e-6,
    expected_reference: ORM | None = None,
    max_candidate_change_percent: float | None = None,
    caget: Callable | None = None,
    caput: Callable | None = None,
) -> WriteResult:
    """Write Rxx/Ryy, verify both, and restore both after any partial failure."""
    # Safety check that will exit function if no ORM of the safetyDecision is provided 
    if isinstance(value, SafetyDecision):
        if not value.should_write or value.orm is None:
            return WriteResult(False, False, False, value.reason)
        orm = value.orm
    else:
        orm = value
    if orm.system != "SOFB":
        raise ValueError("the EPICS writer accepts SOFB matrices only")
    validate_candidate(orm)
    if (not np.isfinite(readback_rtol) or readback_rtol < 0
            or not np.isfinite(readback_atol_nm_per_a) or readback_atol_nm_per_a < 0):
        raise ValueError("readback tolerances must be nonnegative and finite")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("EPICS timeout must be positive and finite")
    if not isinstance(writes_enabled, bool) or not isinstance(enable_writes, bool):
        raise ValueError("both write arms must be booleans")
    if expected_reference is not None and max_candidate_change_percent is None:
        raise ValueError("an expected reference requires a candidate-change limit")
    if not (writes_enabled and enable_writes): # Both of these need to be trure to be able to write to EPICS
        return WriteResult(False, False, False, "dry-run: both write arms are required")

    get = caget or _default_caget()
    put = caput or _default_caput()
    previous = read_current_orm(timeout=timeout, caget=get) # Saves existing ORM in case the write fails and previous ORM needs to be restored
    if expected_reference is not None:
        validate_candidate(expected_reference)
        if (not np.allclose(previous.Rxx, expected_reference.Rxx,
                            rtol=readback_rtol, atol=readback_atol_nm_per_a)
                or not np.allclose(previous.Ryy, expected_reference.Ryy,
                                   rtol=readback_rtol, atol=readback_atol_nm_per_a)):
            raise UnsafeMachineStateError("in-use ORM changed since the reference was selected")
        from .safety import choose_safe_orm
        live_decision = choose_safe_orm(orm, previous,
            max_candidate_change_percent=max_candidate_change_percent)
        if not live_decision.accepted:
            raise UnsafeMachineStateError(f"candidate failed fresh in-use ORM gate: {live_decision.reason}")
    horizontal, vertical = _vectors(orm)
    #writes horizontal ORM first then vertical, waits until each write to be complete before moving on
    try:
        _put(put, config.OFB_H_MATRIX_PV, horizontal, timeout) 
        _put(put, config.OFB_V_MATRIX_PV, vertical, timeout)
        _verify(
            orm,
            caget=get,
            timeout=timeout,
            rtol=readback_rtol,
            atol=readback_atol_nm_per_a,
        )
    except Exception as write_error: # if any exception occurs code moves to here
        # Each plane gets its own restore and readback attempt. A failure on
        # one channel must not strand the other plane on the test matrix.
        old_h, old_v = _vectors(previous)
        planes = ((config.OFB_H_MATRIX_PV, old_h),
                  (config.OFB_V_MATRIX_PV, old_v))
        command_errors = []
        for pv, vector in planes:
            try:
                _put(put, pv, vector, timeout)
            except Exception as exc:
                command_errors.append(f"{pv}: {exc}")
        verification_errors = []
        for pv, vector in planes:
            try:
                actual = _read_vector(pv, vector.size, timeout=timeout, caget=get)
                if not np.allclose(actual, vector, rtol=readback_rtol,
                                   atol=readback_atol_nm_per_a):
                    raise RuntimeError("restored response-matrix readback did not verify")
            except Exception as exc:
                verification_errors.append(f"{pv}: {exc}")
        if verification_errors:
            details = "; ".join(command_errors + verification_errors)
            raise UnsafeMachineStateError(
                f"matrix write failed ({write_error}); restoration also failed "
                f"({details})"
            ) from write_error
        # A timed-out put may have reached the server. Fresh matching readback
        # establishes restoration even when its acknowledgement was lost.
        reason = f"write failed; previous values restored: {write_error}"
        if command_errors:
            reason += "; rollback command warnings: " + "; ".join(command_errors)
        return WriteResult(True, False, True, reason)
    return WriteResult(True, True, False, "write and readback verification succeeded")

# load an ORM from file, validates it, and then return it
def load_verified_reference(path: str | Path) -> ORM | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        result = ORM.load(path)
        validate_candidate(result)
        return result
    except (OSError, ValueError):
        return None
