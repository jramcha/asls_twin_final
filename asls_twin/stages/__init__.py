"""Named stage dispatch for the ASLS ORM configurations."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from ..types import LocoMeasurement, ORM


AVAILABLE_STAGES = (
    "0",
    "1",
    "2a",
    "2h",
    "3h",
    "3h_weighted",
    "6h",
    "7a",
)

# These implementations are committed for bounded offline evaluation. They are
# not selectable by the long-running service without an explicit opt-in.
EXPERIMENTAL_STAGES = frozenset()
CURRENT_STAGES = frozenset(("2h", "3h", "3h_weighted", "6h"))


def normalize_stage(stage: str | int) -> str:
    label = str(stage).strip().lower()
    if label not in AVAILABLE_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return label


def stage_catalog(system="SOFB") -> dict[str, dict[str, bool]]:
    from ..systems import get_system
    get_system(system)
    if system == "FOFB":
        from ..fofb import STAGES
        return {label: {"requires_beam_current": label in CURRENT_STAGES,
                        "experimental": label in EXPERIMENTAL_STAGES} for label in STAGES}
    return {
        label: {
            "requires_beam_current": label in CURRENT_STAGES,
            "experimental": label in EXPERIMENTAL_STAGES,
        }
        for label in AVAILABLE_STAGES
    }


def calculate_stage(
    stage: str | int,
    state: dict[str, float],
    loco_history: list[LocoMeasurement] | None = None,
    *,
    workers: int = 1,
    beam_current_A: float | None = None,
    target_time: datetime | None = None,
    machine_state=None,
    reference_ids: dict[str, float] | None = None,
    remaining_ids: tuple[str, ...] = ("ID04", "ID12"),
    allow_experimental: bool = False,
    stage7_model=None,
    system: str = "SOFB",
    scales=None,
    units: str | None = None,
    fofb_twin=None,
    orm_method=None,
) -> ORM:
    """Validate a stage label and dispatch without performing machine I/O."""
    from ..systems import get_system
    get_system(system)
    if system == "FOFB":
        if machine_state is not None or reference_ids is not None:
            raise ValueError("extended machine state/reference IDs are not supported for FOFB")
        if str(stage).lower() == "7a" and stage7_model is not None:
            if getattr(stage7_model, "system", None) != "FOFB":
                raise ValueError("FOFB requires a FOFB Stage 7 model")
            if units not in (None, "nm/A"):
                raise ValueError("FOFB Stage 7 returns nm/A")
            if any(r.orm.system != "FOFB" for r in (loco_history or [])):
                raise ValueError("FOFB history cannot contain SOFB measurements")
            return stage7_model.predict(state).check()
        from ..fofb import FOFBTwin
        if fofb_twin is not None and (scales is not None or orm_method is not None):
            raise ValueError("set scales/method on FOFBTwin or the call, not both")
        twin = fofb_twin if fofb_twin is not None else FOFBTwin(scales=scales, orm_method=orm_method or "linear")
        return twin.calculate(stage, state, loco_history, workers=workers,
                              beam_current_A=beam_current_A, target_time=target_time,
                              units=units, allow_experimental=allow_experimental)
    if (fofb_twin is not None or scales is not None or units not in (None, "nm/A")
            or orm_method not in (None, "bipolar")):
        raise ValueError("SOFB stage dispatch uses its existing units and calibration")
    if any(r.orm.system != "SOFB" for r in (loco_history or [])):
        raise ValueError("SOFB history cannot contain FOFB measurements")
    if stage7_model is not None and getattr(stage7_model, "system", "SOFB") != "SOFB":
        raise ValueError("SOFB requires a SOFB Stage 7 model")
    label = normalize_stage(stage)
    if label in EXPERIMENTAL_STAGES and not allow_experimental:
        raise ValueError(f"Stage {label} requires explicit experimental opt-in")
    module = importlib.import_module(f"{__name__}.stage{label}")
    history = loco_history if loco_history is not None else []
    result = module.calculate(
        state,
        history,
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
        machine_state=machine_state,
        reference_ids=reference_ids,
        remaining_ids=remaining_ids,
        **({"stage7_model": stage7_model} if label == "7a" else {}),
    )
    if result.system != "SOFB":
        raise ValueError("SOFB stage returned a matrix for another system")
    return result
