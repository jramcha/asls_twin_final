"""Typed values shared by the production twin."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from . import config


@dataclass(frozen=True)
class ORM:
    """Four response blocks with explicit units and feedback-system identity."""

    Rxx: np.ndarray
    Ryx: np.ndarray
    Rxy: np.ndarray
    Ryy: np.ndarray
    units: str
    system: str = "SOFB"

    def check(self, *, finite: bool = True) -> "ORM":
        from .systems import get_system
        expected = get_system(self.system).shapes
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(
                    f"{self.system} block {name} has shape {value.shape}, expected {shape}"
                )
            if finite and not np.isfinite(value).all():
                raise ValueError(f"{self.system} block {name} contains non-finite values")
        return self

    @property
    def upload_vector(self) -> np.ndarray:
        """Rxx then Ryy in C order; this is not a FOFB controller wire format."""
        self.check()
        return np.concatenate((self.Rxx.ravel(order="C"), self.Ryy.ravel(order="C")))

    def copy(self) -> "ORM":
        return ORM(
            self.Rxx.copy(),
            self.Ryx.copy(),
            self.Rxy.copy(),
            self.Ryy.copy(),
            self.units,
            self.system,
        )

    def save(self, path: str | Path) -> Path:
        self.check()
        from .systems import get_system
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            Rxx=self.Rxx,
            Ryx=self.Ryx,
            Rxy=self.Rxy,
            Ryy=self.Ryy,
            units=np.array(self.units),
            system=np.array(self.system),
            contract=np.array(get_system(self.system).fingerprint),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ORM":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                np.asarray(data["Rxx"], float),
                np.asarray(data["Ryx"], float),
                np.asarray(data["Rxy"], float),
                np.asarray(data["Ryy"], float),
                str(data["units"]),
                str(data["system"]),
            )
            if result.system == "FOFB":
                from .systems import get_system
                if "contract" not in data or str(data["contract"]) != get_system("FOFB").fingerprint:
                    raise ValueError("FOFB artifact device/unit contract is missing or incompatible")
        return result.check()


@dataclass(frozen=True)
class LocoMeasurement:
    path: Path
    timestamp: datetime
    file_hash: str
    orm: ORM
    id_state: Mapping[str, float] | None = None
    id_state_source: str | None = None
    beam_current_A: float | None = None


@dataclass(frozen=True)
class SafetyDecision:
    orm: ORM | None
    accepted: bool
    should_write: bool
    reason: str
    change_percent: float | None = None


@dataclass(frozen=True)
class WriteResult:
    attempted: bool
    verified: bool
    restored: bool
    reason: str


@dataclass
class UpdateTracker:
    last_update_monotonic: float | None = None
    accepted_id_strengths: np.ndarray | None = None
    accepted_orm: ORM | None = None
    metadata: dict[str, object] = field(default_factory=dict)
