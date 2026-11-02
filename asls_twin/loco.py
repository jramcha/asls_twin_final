# Last reviewed: 19/09/2026
"""Discovery, validation, and ID-state association for weekly LOCO files."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np
import scipy.io as sio

from . import config
from .epics import read_id_state
from .loco_quality import accepted_history_current
from .safety import validate_id_state
from .types import LocoMeasurement, ORM
from .systems import get_system

# Use acquisition time encoded in the measurement name for history ordering.
_FILENAME = re.compile(
    r"^BPMRespMat_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.mat$"
)


class IneligibleIDState(ValueError):
    """A measured ID readback is unusable by the validated twin model."""


def parse_loco_timestamp(path: str | Path) -> datetime:
    match = _FILENAME.match(Path(path).name)
    if match is None:
        raise ValueError(f"invalid LOCO filename {Path(path).name!r}")
    return datetime(*[int(value) for value in match.groups()])

# Search subdirectories.
def discover_loco_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"LOCO directory does not exist: {directory}")
    result = [path for path in directory.rglob("BPMRespMat_*.mat") if path.is_file()]
    for path in result:
        parse_loco_timestamp(path)
    return sorted(result, key=parse_loco_timestamp)

# Hash file contents in chunks
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Read the four MATLAB response blocks and the recorded current. 
def _load_payload(path: Path, *, strict_metadata: bool = False, system="SOFB") -> tuple[ORM, float | None]:
    spec = get_system(system)
    try:
        data = sio.loadmat(
            path, squeeze_me=True, struct_as_record=False, variable_names=["Rmat"]
        )
    except Exception as exc:
        raise ValueError(f"cannot load LOCO file {path}: {exc}") from exc
    if set(data).isdisjoint({"Rmat"}):
        raise ValueError(f"LOCO file has no Rmat: {path}")
    rmat = data["Rmat"]
    if not isinstance(rmat, np.ndarray) or rmat.shape != (2, 2):
        raise ValueError(f"Rmat in {path} has shape {getattr(rmat, 'shape', None)}, expected (2, 2)")
    # Check the monitor and actuator lists against AO before interpreting rows and columns.
    if strict_metadata or system == "FOFB":
        for row in range(2):
            for col in range(2):
                block = rmat[row, col]
                if str(getattr(block, "UnitsString", "")) not in {"nm/ampere", "nm/A"}:
                    raise ValueError(f"unverified ORM units in {path}")
                bpm = config.AO.family(("BPMx", "BPMy")[row]).devices
                family = (spec.horizontal_family, spec.vertical_family)[col]
                cor = config.AO.family(family).devices
                if system == "FOFB":
                    if str(getattr(getattr(block, "Actuator", None), "FamilyName", "")) != family:
                        raise ValueError(f"unverified FOFB actuator family in {path}")
                    if str(getattr(getattr(block, "Monitor", None), "FamilyName", "")) != ("BPMx", "BPMy")[row]:
                        raise ValueError(f"unverified FOFB BPM family in {path}")
                for obj, expected in ((getattr(block, "Monitor", None), bpm),
                                      (getattr(block, "Actuator", None), cor)):
                    if not np.array_equal(getattr(obj, "DeviceList", None), expected):
                        raise ValueError(f"ORM device ordering differs from MML in {path}")
    try:
        result = ORM(
            np.asarray(rmat[0, 0].Data, float),
            np.asarray(rmat[1, 0].Data, float),
            np.asarray(rmat[0, 1].Data, float),
            np.asarray(rmat[1, 1].Data, float),
            "nm/A",
            system,
        )
    except Exception as exc:
        raise ValueError(f"Rmat in {path} does not contain the four Data blocks") from exc
    result.check(finite=True)
    raw_current = getattr(rmat[0, 0], "DCCT", None)
    if raw_current is None:
        current = None
    else:
        values = np.asarray(raw_current, float).ravel()
        if values.shape != (1,) or not np.isfinite(values[0]) or not 0 < values[0] <= 1000:
            raise ValueError(f"invalid historical DCCT in {path}")
        current = float(values[0]) / 1000.0
    return result, current

# The general history validation needs only the matrix, using the default
# SOFB path.
def _load_orm(path: Path) -> ORM:
    return _load_payload(path)[0]


def _known_missing_bpm_history(path: Path) -> bool:
    """Recognize SOFB ORMs missing only BPM device (3, 3)."""
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False,
                       variable_names=["Rmat"])
    blocks = data.get("Rmat")
    if not isinstance(blocks, np.ndarray) or blocks.shape != (2, 2):
        return False
    for row, bpm_family in enumerate(("BPMx", "BPMy")):
        expected = np.asarray(config.AO.family(bpm_family).devices)
        expected = expected[~np.all(expected == (3, 3), axis=1)]
        for col, ncor in enumerate((config.N_HCM, config.N_VCM)):
            block = blocks[row, col]
            if np.shape(getattr(block, "Data", None)) != (len(expected), ncor):
                return False
            actual = getattr(getattr(block, "Monitor", None), "DeviceList", None)
            if not np.array_equal(actual, expected):
                return False
    return True


def _fofb_response(path: Path) -> bool:
    """Identify HFC/VFC response files before SOFB history validation."""
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False,
                       variable_names=["Rmat"])
    blocks = data.get("Rmat")
    if not isinstance(blocks, np.ndarray) or blocks.shape != (2, 2):
        return False
    for row in range(2):
        for col, family in enumerate(("HFC", "VFC")):
            actuator = getattr(blocks[row, col], "Actuator", None)
            name = str(getattr(actuator, "FamilyName", ""))
            devices = getattr(actuator, "DeviceList", None)
            if name != family and not np.array_equal(
                    devices, config.AO.family(family).devices):
                return False
    return True

# Accept either stem.id_state.json or filename.mat.id_state.json beside
# the measurement.
def _sidecar_paths(path: Path) -> tuple[Path, Path]:
    return (path.with_suffix(".id_state.json"), Path(str(path) + ".id_state.json"))

# Read recorded ID metadata from an adjacent JSON file.
def _sidecar_state(path: Path, digest: str) -> dict[str, float] | None:
    existing = [candidate for candidate in _sidecar_paths(path) if candidate.exists()]
    if len(existing) > 1:
        raise ValueError(f"multiple ID-state sidecars exist for {path}")
    if not existing:
        return None
    try:
        value = json.loads(existing[0].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ID-state sidecar {existing[0]}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"ID-state sidecar {existing[0]} must contain an object")
    if "file_sha256" in value and value["file_sha256"] != digest:
        raise ValueError(f"ID-state sidecar hash does not match {path}")
    state = value.get("id_state", value)
    if not isinstance(state, dict):
        raise ValueError(f"ID-state sidecar {existing[0]} has no id_state object")
    try:
        return validate_id_state(state)
    except ValueError as exc:
        raise IneligibleIDState(f"invalid recorded ID state in {existing[0]}: {exc}") from exc

# The registry maps measurement-content hashes to their associated metadata.
def _load_registry(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid LOCO registry {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LOCO registry {path} must contain an object")
    return value


def _fofb_lhs_state(path):
    """Read acquisition-time ID values from the collector-owned run record."""
    record_path = path.parent / "record.mat"
    if not record_path.exists():
        return None
    data = sio.loadmat(record_path, simplify_cells=True)
    if (data.get("measurementComplete") != 1
            or data.get("measurementSkipped", 0) != 0
            or data.get("idStateChangedDuringMeasurement") != 0):
        raise ValueError("FOFB LHS record is incomplete, skipped, or has moving IDs")
    ids = np.asarray(data.get("ids", []), float).ravel()
    expected = np.array([int(name[2:]) for name in config.ID_NAMES])
    if not np.array_equal(ids, expected):
        raise ValueError("FOFB LHS record has an incompatible ID order")
    before = data.get("before", {})
    values = np.asarray(before.get("ids", []), float).ravel()
    if values.shape != expected.shape:
        raise ValueError("FOFB LHS record has no complete pre-measurement ID readbacks")
    return validate_id_state(dict(zip(config.ID_NAMES, values)))

# Replace the registry only after the complete JSON has been written to a
# temporary file.
def _write_registry(path: Path, value: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


# Review synthesis (Nick): the acquisition timestamp comes from the filename;
# registry entries use the content hash; sidecar hashes are checked when supplied.
def load_loco_file(
    path: str | Path,
    *,
    registry: dict[str, dict] | None = None,
    strict_metadata: bool = False,
    system="SOFB",
) -> LocoMeasurement:
    path = Path(path)
    timestamp = parse_loco_timestamp(path)
    digest = file_sha256(path)
    sidecar = _sidecar_state(path, digest)
    entry = {} if registry is None else registry.get(digest, {})
    # Older service versions inferred this source from file mtime. Those
    # entries cannot establish acquisition settings and must not be reused.
    registry_value = (None if entry.get("source") == "strict_live_readback"
                      else entry.get("id_state"))
    if sidecar is not None:
        state, source = sidecar, "sidecar"
    elif registry_value is not None:
        state, source = validate_id_state(registry_value), "registry"
    else:
        state, source = None, None
    if system == "FOFB":
        recorded = _fofb_lhs_state(path)
        if recorded is not None:
            if state is not None and state != recorded:
                raise ValueError("FOFB sidecar/registry disagrees with acquisition ID readbacks")
            state, source = recorded, "fofb_lhs_record"
    orm, beam_current_A = _load_payload(path, strict_metadata=strict_metadata, system=system)
    return LocoMeasurement(
        path, timestamp, digest, orm, state, source, beam_current_A
    )

# Read SOFB history. Discovery records missing provenance but never assigns
# current live settings to a measurement made at an earlier time.
def read_loco_history(
    directory: str | Path,
    *,
    registry_file: str | Path | None = None,
    now: datetime | None = None,
    max_age_seconds: float | None = None,
    complete_grace_seconds: float = 0,
    capture_window_seconds: float = 300,
    capture_new: bool = False,
    state_reader: Callable[[], dict[str, float]] = read_id_state,
) -> list[LocoMeasurement]:
    paths = discover_loco_files(directory)
    if not paths:
        raise FileNotFoundError(f"no BPMRespMat_*.mat files under {directory}")
    now = now or datetime.now()
    # Historical acceptance is shared by every SOFB stage. Raw measurements
    # stay on disk, including low-current LHS runs and incomplete ORMs.
    valid_paths = []
    low_current = 0
    for path in paths:
        try:
            _, current_A = _load_payload(path, strict_metadata=True)
        except ValueError:
            if _known_missing_bpm_history(path):
                logging.getLogger(__name__).warning(
                    "Skipping incomplete LOCO ORM %s: BPM (3, 3) was not measured", path)
                continue
            if _fofb_response(path):
                logging.getLogger(__name__).info(
                    "Skipping FOFB ORM %s from SOFB history", path)
                continue
            raise
        if not accepted_history_current(current_A):
            low_current += 1
            continue
        try:
            _sidecar_state(path, file_sha256(path))
        except IneligibleIDState as exc:
            logging.getLogger(__name__).warning("Skipping LOCO %s: %s", path, exc)
            continue
        valid_paths.append(path)
    if low_current:
        logging.getLogger(__name__).info(
            "Excluded %d LOCO measurements below 170 mA or without valid current", low_current)
    if not valid_paths:
        raise ValueError("no complete LOCO ORM files in history")
    paths = valid_paths
    newest_timestamp = parse_loco_timestamp(paths[-1])
    age = (now - newest_timestamp).total_seconds()
    if age < 0:
        raise ValueError("newest LOCO timestamp is in the future")
    if max_age_seconds is not None and age > max_age_seconds:
        raise RuntimeError(
            f"newest LOCO measurement is stale ({age:.0f} s > {max_age_seconds:.0f} s)"
        )
    # Leave a recently modified file alone until its completion grace expires.
    # This reduces the chance of reading while the collector is still writing.
    if (
        complete_grace_seconds > 0
        and time.time() - paths[-1].stat().st_mtime < complete_grace_seconds
    ):
        raise RuntimeError("newest LOCO file is still inside the completion grace period")

    registry_path = Path(registry_file) if registry_file is not None else None
    registry = _load_registry(registry_path) if registry_path is not None else {}
    newest_hash = file_sha256(paths[-1])
    if capture_new and newest_hash not in registry:
        # Files may be copied or renamed long after acquisition. Neither mtime
        # nor discovery time proves that the current ID settings produced them.
        # An acquisition sidecar is the only way to attach those settings.
        registry[newest_hash] = {
            "path": str(paths[-1].resolve()),
            "timestamp": newest_timestamp.isoformat(),
            "captured_at": now.isoformat(),
            "id_state": None,
            "capture_error": "no acquisition-time ID sidecar; live state was not associated",
        }
        if registry_path is not None:
            _write_registry(registry_path, registry)
    history = []
    by_hash = {}
    for path in paths:
        record = load_loco_file(path, registry=registry, strict_metadata=True)
        earlier = by_hash.get(record.file_hash)
        if earlier is not None:
            if (earlier.timestamp != record.timestamp
                    or (earlier.id_state is not None and record.id_state is not None
                        and earlier.id_state != record.id_state)):
                raise ValueError(f"conflicting LOCO metadata for duplicate ORM {path}")
            if earlier.id_state is None and record.id_state is not None:
                history[history.index(earlier)] = record
                by_hash[record.file_hash] = record
            continue
        by_hash[record.file_hash] = record
        history.append(record)
    return history

# Reuse the same validation and capture rules, then return the last entry
# from the acquisition-time-sorted history.
def read_latest_loco(
    directory: str | Path = "data/loco",
    *,
    registry_file: str | Path | None = None,
    now: datetime | None = None,
    max_age_seconds: float | None = None,
    complete_grace_seconds: float = 0,
    capture_window_seconds: float = 300,
    capture_new: bool = False,
    state_reader: Callable[[], dict[str, float]] = read_id_state,
) -> LocoMeasurement:
    return read_loco_history(
        directory,
        registry_file=registry_file,
        now=now,
        max_age_seconds=max_age_seconds,
        complete_grace_seconds=complete_grace_seconds,
        capture_window_seconds=capture_window_seconds,
        capture_new=capture_new,
        state_reader=state_reader,
    )[-1]


# Review synthesis (Nick, adapted to active Stage 7): a valid training record
# needs recorded ID settings and compatible units/device ordering. The reader
# does not substitute current live ID values for a historical acquisition.
class Stage7HistoryReader:
    """Cache unchanged, validated records; an invalid arrival cannot poison history.

    No discovery-time live ID association: training requires recorded settings.
    The service controls cadence, including retries of incomplete files.
    """

    def __init__(self, directory, registry_file, training_start=None, grace_seconds=30, *, system="SOFB"):
        get_system(system)
        self.system = system
        self.directories = (tuple(Path(item) for item in directory)
                            if isinstance(directory, (list, tuple)) else (Path(directory),))
        if not self.directories:
            raise ValueError("Stage 7 requires at least one history directory")
        self.registry_file = Path(registry_file)
        self.training_start = training_start
        self.grace_seconds = grace_seconds
        self.cache = {}

    # Track size and nanosecond modification/change times for cache validation.
    @staticmethod
    def _stamp(path):
        if not path.exists():
            return None
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def read(self, *, now=None):
        now = now or datetime.now()
        registry = _load_registry(self.registry_file)
        registry_stamp = self._stamp(self.registry_file)
        for directory in self.directories:
            if not directory.is_dir():
                raise FileNotFoundError(f"LOCO directory does not exist: {directory}")
        history, retained, seen_hashes = [], {}, {}
        paths = sorted({path for directory in self.directories
                        for path in directory.rglob("BPMRespMat_*.mat")})
        for path in paths:
            try:
                timestamp = parse_loco_timestamp(path)
                if timestamp > now or (self.training_start is not None and timestamp < self.training_start):
                    continue
                if time.time() - path.stat().st_mtime < self.grace_seconds:
                    continue
                dependencies = (path, *_sidecar_paths(path))
                if self.system == "FOFB":
                    dependencies += (path.parent / "record.mat",)
                stamp = tuple(self._stamp(p) for p in dependencies) + (registry_stamp,)
                cached = self.cache.get(path)
                if cached is not None and cached[0] == stamp:
                    record = cached[1]
                else:
                    record = load_loco_file(path, registry=registry, strict_metadata=True, system=self.system)
                    after = tuple(self._stamp(p) for p in dependencies) + (self._stamp(self.registry_file),)
                    if after != stamp:
                        raise ValueError("LOCO or ID metadata changed while being read")
                if not accepted_history_current(record.beam_current_A):
                    continue
                if record.id_state is None:
                    raise ValueError("no recorded ID settings; not eligible for LOCO history")
                previous = seen_hashes.get(record.file_hash)
                if previous is not None:
                    if previous.timestamp != record.timestamp or previous.id_state != record.id_state:
                        raise ValueError("conflicting metadata for duplicate LOCO ORM")
                    retained[path] = (stamp, record)
                    continue
                retained[path] = (stamp, record)
                seen_hashes[record.file_hash] = record
                history.append(record)
            # Skip an invalid arrival without discarding other valid measurements.
            except (ValueError, TypeError, AttributeError, OSError, RuntimeError) as exc:
                logging.getLogger(__name__).warning("Skipping LOCO %s: %s", path, exc)
        self.cache = retained
        return sorted(history, key=lambda r: r.timestamp)
