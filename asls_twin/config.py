"""Configuration loading and machine constants."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Backport for facility Python 3.9/3.10.
import math
import json
import os

from scipy.io import whosmat

from .mml import MMLAO


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"

LATTICE_MAT = DATA / "assr4_splitbends_modified_AT_SIMPLE.mat"
M66_FITS_MAT = DATA / "allIDs_M66fits_2026-08-27.mat"
QUAD_FF_MAT = DATA / "allIDs_quadFF_2026-08-27.mat"
REGRESSION_ARTIFACT = DATA / "scan_ridge_v1.npz"
# The frozen regression file is accepted only when its bytes match this
# known artifact hash. A different fit needs an explicitly updated reference.
REGRESSION_ARTIFACT_SHA256 = "69ac051f1e85850f8e4b7ad64e3cac11999a7bc1181ee63e99ab060f5dfe8fa4"

# AO owns hardware metadata. This companion reference holds only values that
# are absent from AO (published model, fitted calibrations, field/output PVs).
MODEL_REFERENCE_FILE = DATA / "model_reference.json"
MODEL_REFERENCE = json.loads(MODEL_REFERENCE_FILE.read_text())
ENERGY_EV = float(MODEL_REFERENCE["energy_ev"])
BRHO = float(MODEL_REFERENCE["brho_tm"])
DEFAULT_MACHINE_MODE = int(MODEL_REFERENCE["default_machine_mode"])
MODE_SETTINGS = {int(k): v for k, v in MODEL_REFERENCE["mode_settings"].items()}
ID_NAMES = tuple(MODEL_REFERENCE["id_names"])
GAP_FEATURE_PERIOD_MM = MODEL_REFERENCE["gap_feature_period_mm"]
FIELD_SQUARED_IDS = frozenset(MODEL_REFERENCE["field_pvs"])
REGRESSION_FEATURE_VERSION = MODEL_REFERENCE["regression_feature_version"]
# These are the deployed SOFB conversion factors for the two diagonal
# response blocks.
OFB_RXX_SCALE = float(MODEL_REFERENCE["ofb_scales"]["Rxx"])
OFB_RYY_SCALE = float(MODEL_REFERENCE["ofb_scales"]["Ryy"])
LOCO_FIT_SCALE_CANDIDATES = {int(k): tuple(v) for k, v in MODEL_REFERENCE["loco_fit_scale_candidates"].items()}
DEPLOYED_ORM_REFERENCE_MAT = DATA / "response_sofb_system_20260819.mat"
ID_READBACK_RANGES = {k: tuple(v) for k, v in MODEL_REFERENCE["id_readback_ranges"].items()}
FIELD_MAX_T = MODEL_REFERENCE["field_max_t"]
GAP_FIELD_TABLES = {k: DATA / v for k, v in MODEL_REFERENCE["gap_field_tables"].items()}
GAP_FIELD_ENDPOINTS = {k: tuple(v) for k, v in MODEL_REFERENCE["gap_field_endpoints"].items()}
OFB_H_MATRIX_PV = MODEL_REFERENCE["output_pvs"]["Rxx"]
OFB_V_MATRIX_PV = MODEL_REFERENCE["output_pvs"]["Ryy"]
CORRECTOR_LATTICE_FAMILIES = MODEL_REFERENCE["corrector_lattice_families"]

# Numerical/test policies, not AO hardware properties.
KICK_RAD = 1e-5
HISTORY_COUNT = 3
# Stage 3h only; Stage 6h retains the policy used to train its frozen ridge.
STAGE3H_HISTORY_HALF_LIFE_DAYS = 90.0
STAGE3H_HISTORY_ID_BANDWIDTH = 1.0
STAGE3H_HISTORY_CURRENT_BANDWIDTH = 1.0
STAGE3H_HISTORY_SPAN_FLOOR_FRACTION = 0.10
STAGE3H_HISTORY_SPAN_CEILING_FRACTION = 0.40
STAGE3H_HISTORY_SCALE_FLOOR = 1e-9
STAGE3H_HISTORY_CURRENT_SCALE_FLOOR_A = 0.020  # 20 mA

DEFAULT_EPICS_TIMEOUT = 2.0
LIVE_ID_CHANGE_TOLERANCE = {"field": 0.001, "gap": 0.01}
LIVE_BEAM_CURRENT_MAX_A = 1.0

# Use the packaged AO as the reference for device ordering and supported units
PACKAGED_AO_MML = DATA / "MML_AO.mat"
_PACKAGED_AO = MMLAO(PACKAGED_AO_MML)


def read_mml(path: str | Path) -> MMLAO:
    """Read and validate an export without changing the active configuration."""
    return MMLAO(path).validate(_PACKAGED_AO)


def configure_mml(path: str | Path) -> MMLAO:
    """Select a validated AO at startup; no hardware or model computation.

    The packaged device/unit contract must match so already-loaded ORM classes,
    trained artifacts and lattice caches cannot silently acquire new row meanings.
    All validation finishes before publishing the new configuration.
    """
    ao = read_mml(path)
    id_family = ao.family("ID")
    sectors = {}
    marker_indexes = {}
    channels = {}
    # Locate each model ID by its sector/device identity in AO, then record
    # its position within the ID family for later lattice-marker lookup.
    for name in ID_NAMES:
        matches = [device for device in id_family.devices if device[0] == int(name[2:])]
        if len(matches) != 1:
            raise ValueError(f"AO must have exactly one device for model ID {name}")
        device = matches[0]
        sectors[name] = device[0]
        marker_indexes[name] = id_family.devices.index(device)
        channels[name] = id_family.channel(device)
    ids = {name: (MODEL_REFERENCE["field_pvs"][name], "field")
           if name in FIELD_SQUARED_IDS else (channels[name], "gap") for name in ID_NAMES}
    # Fit keys refer to physical device identities, not the spelling of a PV.
    # Match artifact keys to AO sector/device identities; never to Status flags.
    def fit_keys(path):
        available = {name for name, _, _ in whosmat(path)}
        identities = {f"ID{sector:02d}": f"SR{sector:02d}ID{device:02d}"
                      for sector, device in id_family.devices}
        return {name: identities[name] for name in ID_NAMES if identities[name] in available}
    counts = {k: v.count for k, v in ao.families.items()}
    quads, sexts = ao.members("QUAD"), ao.members("SEXT")
    expected = {name: counts[name] for name in quads + sexts}
    expected.update(ID=counts["ID"], BPM=counts["BPMx"])
    globals().update(
        AO=ao, AO_MML_FILE=ao.path,
        N_BPM=counts["BPMx"], N_HCM=counts["HCM"], N_VCM=counts["VCM"],
        N_ID_MARKERS=counts["ID"], FAMILY_COUNTS=counts,
        QUAD_FAMILIES=quads, SEXT_FAMILIES=sexts, EXPECTED_FAMILY_COUNTS=expected,
        ID_PVS=ids, ID_SECTOR=sectors, ID_MARKER_INDEX=marker_indexes, M66_KEY=fit_keys(M66_FITS_MAT),
        QUAD_FF_KEY=fit_keys(QUAD_FF_MAT),
        BEAM_CURRENT_PV=ao.family("DCCT").channels[0],
        BEAM_CURRENT_HW_TO_A=float(ao.linear_scale("DCCT")[0]),
    )
    return ao

# Initialize metadata on import, using the environment-selected AO if set.
configure_mml(os.environ.get("ASLS_TWIN_AO_MML", PACKAGED_AO_MML))

@dataclass(frozen=True)
class UpdateConfig:
    mode: str
    poll_seconds: float
    time_period_seconds: float
    id_frob_trigger_percent: float
    id_frob_cubed_trigger_percent: float
    orm_prediction_trigger_percent: float
    max_push_hz: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_push_hz) or self.max_push_hz <= 0:
            raise ValueError("update.max_push_hz must be positive and finite")

    @property
    def selected_threshold(self) -> float:
        key = {
            "time": "time_period_seconds",
            "idFrob": "id_frob_trigger_percent",
            "idFrobCubed": "id_frob_cubed_trigger_percent",
            "ormPredFrob": "orm_prediction_trigger_percent",
        }.get(self.mode)
        if key is None:
            raise ValueError(f"unsupported update mode {self.mode!r}")
        value = float(getattr(self, key))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"update.{key} must be positive and finite")
        return value

# The candidate-change limit gates a proposed matrix. The two readback
# tolerances are used later to verify what the EPICS writer actually stored.
@dataclass(frozen=True)
class SafetyConfig:
    max_candidate_change_percent: float
    readback_rtol: float
    readback_atol_nm_per_a: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_candidate_change_percent) or self.max_candidate_change_percent <= 0:
            raise ValueError("safety.max_candidate_change_percent must be positive and finite")
        for key in ("readback_rtol", "readback_atol_nm_per_a"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"safety.{key} must be nonnegative and finite")

# Group the measurement location and service output files.
@dataclass(frozen=True)
class PathsConfig:
    loco_directory: Path
    registry_file: Path
    reference_file: Path
    reference_metadata_file: Path
    event_log: Path
    residual_cache_file: Path

# An unset training start allows all otherwise eligible history. A supplied
# start excludes earlier acquisitions from the active Stage 7 fit.
@dataclass(frozen=True)
class Stage7Config:
    training_start: datetime | None = None
    history_directories: tuple[Path, ...] = ()

# Pass one resolved configuration into the service.
@dataclass(frozen=True)
class RuntimeConfig:
    stage: str
    writes_enabled: bool
    workers: int
    loco_max_age_seconds: float
    loco_complete_grace_seconds: float
    loco_capture_window_seconds: float
    update: UpdateConfig
    safety: SafetyConfig
    paths: PathsConfig
    source: Path
    ao_mml_file: Path = PACKAGED_AO_MML
    loco_poll_seconds: float = 600.0
    stage7: Stage7Config = Stage7Config()
    stage7_model_file: Path | None = None


def _table(section: dict, name: str) -> dict:
    value = section.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _resolve(base: Path, value: object, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{key} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()

# Read and check service settings once at startup, then return typed values
# and resolved paths. Also selects the requested AO after validating it.
def load_config(path: str | Path = "config.toml") -> RuntimeConfig:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    service = _table(raw, "service")
    update_raw = _table(raw, "update")
    safety_raw = _table(raw, "safety")
    paths_raw = _table(raw, "paths")

    update = UpdateConfig(
        mode=str(update_raw.get("mode", "")),
        poll_seconds=float(update_raw.get("poll_seconds", 0)),
        time_period_seconds=float(update_raw.get("time_period_seconds", 0)),
        id_frob_trigger_percent=float(update_raw.get("id_frob_trigger_percent", 0)),
        id_frob_cubed_trigger_percent=float(
            update_raw.get("id_frob_cubed_trigger_percent", 0)
        ),
        orm_prediction_trigger_percent=float(
            update_raw.get("orm_prediction_trigger_percent", 0)
        ),
        max_push_hz=float(update_raw.get("max_push_hz", 1.0)),
    )
    if not math.isfinite(update.poll_seconds) or update.poll_seconds <= 0:
        raise ValueError("update.poll_seconds must be positive and finite")
    update.selected_threshold

    # Import stage dispatch after module configuration is initialized; stage
    # modules also depend on this configuration module.
    from .stages import normalize_stage

    stage = normalize_stage(service.get("stage", "3b"))

    safety = SafetyConfig(
        max_candidate_change_percent=float(
            safety_raw.get("max_candidate_change_percent", 0)
        ),
        readback_rtol=float(safety_raw.get("readback_rtol", 1e-9)),
        readback_atol_nm_per_a=float(
            safety_raw.get("readback_atol_nm_per_a", 1e-6)
        ),
    )

    # Resolve all service files from the same configuration directory
    base = path.parent
    paths = PathsConfig(
        loco_directory=_resolve(base, paths_raw.get("loco_directory"), "loco_directory"),
        registry_file=_resolve(base, paths_raw.get("registry_file"), "registry_file"),
        reference_file=_resolve(base, paths_raw.get("reference_file"), "reference_file"),
        reference_metadata_file=_resolve(
            base, paths_raw.get("reference_metadata_file"), "reference_metadata_file"
        ),
        event_log=_resolve(base, paths_raw.get("event_log"), "event_log"),
        residual_cache_file=_resolve(
            base,
            paths_raw.get("residual_cache_file", "state/residual_cache.npz"),
            "residual_cache_file",
        ),
    )

    workers = int(service.get("workers", 1))
    if workers < 1:
        raise ValueError("service.workers must be at least one")
    max_age = float(service.get("loco_max_age_seconds", 0))
    grace = float(service.get("loco_complete_grace_seconds", 0))
    capture_window = float(service.get("loco_capture_window_seconds", 0))
    if (not all(map(math.isfinite, (max_age, grace, capture_window)))
            or max_age <= 0 or grace < 0 or capture_window <= grace):
        raise ValueError(
            "LOCO age must be positive and capture window must exceed completion grace"
        )

    loco_poll = float(service.get("loco_poll_seconds", 600))
    if not math.isfinite(loco_poll) or loco_poll <= 0:
        raise ValueError("service.loco_poll_seconds must be positive and finite")
    stage7_raw = raw.get("stage7", {})
    if not isinstance(stage7_raw, dict):
        raise ValueError("[stage7] must be a table")
    start = stage7_raw.get("training_start", "")
    if not isinstance(start, str):
        raise ValueError("stage7.training_start must be a quoted ISO date or datetime")
    try:
        training_start = datetime.fromisoformat(start) if start else None
    except ValueError as exc:
        raise ValueError("stage7.training_start must be an ISO date or datetime") from exc
    if training_start is not None and training_start.tzinfo is not None:
        raise ValueError("stage7.training_start uses local machine time, without a timezone")
    history_roots = stage7_raw.get("history_directories", [])
    if (not isinstance(history_roots, list) or
            any(not isinstance(root, str) or not root for root in history_roots)):
        raise ValueError("stage7.history_directories must be a list of non-empty paths")
    history_directories = tuple(_resolve(base, root, "history_directories")
                                for root in history_roots)

    machine = raw.get("machine", {})
    if not isinstance(machine, dict):
        raise ValueError("[machine] must be a table")
    ao_path = (_resolve(base, machine["ao_mml"], "ao_mml")
               if "ao_mml" in machine else AO_MML_FILE)
    configure_mml(ao_path)

    writes_enabled = service.get("writes_enabled", False)
    if not isinstance(writes_enabled, bool):
        raise ValueError("service.writes_enabled must be a TOML boolean")

    return RuntimeConfig(
        stage=stage,
        writes_enabled=writes_enabled,
        workers=workers,
        loco_max_age_seconds=max_age,
        loco_complete_grace_seconds=grace,
        loco_capture_window_seconds=capture_window,
        update=update,
        safety=safety,
        paths=paths,
        source=path,
        ao_mml_file=AO_MML_FILE,
        loco_poll_seconds=loco_poll,
        stage7=Stage7Config(training_start, history_directories),
        stage7_model_file=_resolve(base, paths_raw.get("stage7_model_file", "state/stage7_active.npz"), "stage7_model_file"),
    )
