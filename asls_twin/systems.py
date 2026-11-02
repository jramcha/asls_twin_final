"""Device contracts for offline ORM generation; no controller addresses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from . import config


@dataclass(frozen=True)
class FeedbackSystem:
    name: str
    horizontal_family: str
    vertical_family: str
    horizontal_selection: tuple
    vertical_selection: tuple
    n_bpm: int
    n_horizontal: int
    n_vertical: int

    @property
    def shapes(self):
        return {"Rxx": (self.n_bpm, self.n_horizontal),
                "Ryx": (self.n_bpm, self.n_horizontal),
                "Rxy": (self.n_bpm, self.n_vertical),
                "Ryy": (self.n_bpm, self.n_vertical)}

    @property
    def fingerprint(self):
        payload = dict(name=self.name, shapes=self.shapes,
                       selections=(self.horizontal_selection, self.vertical_selection),
                       devices={name: config.AO.family(name).devices for name in
                                ("BPMx", "BPMy", self.horizontal_family, self.vertical_family)},
                       units="nm/A", schema="feedback-system-v1")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_system(system="SOFB") -> FeedbackSystem:
    """Resolve an explicit name, without changing process-wide configuration."""
    if system == "SOFB":
        families = config.CORRECTOR_LATTICE_FAMILIES
        return FeedbackSystem("SOFB", "HCM", "VCM",
                              tuple((name, 1, 0) for name in families["HCM"]),
                              tuple((name, 1, 0) for name in families["VCM"]),
                              config.N_BPM, config.N_HCM, config.N_VCM)
    if system != "FOFB":
        raise ValueError(f"unknown feedback system {system!r}; choose SOFB or FOFB")
    for name in ("BPMx", "BPMy", "HFC", "VFC"):
        actual, expected = config.AO.family(name), config._PACKAGED_AO.family(name)
        if (actual.devices != expected.devices or actual.hw_units != expected.hw_units
                or actual.physics_units != expected.physics_units):
            raise ValueError(f"AO.{name} differs from the packaged FOFB device/unit contract")
    return FeedbackSystem("FOFB", "HFC", "VFC",
                          (("SDA", 1, 0), ("SDB", 2, 0)),
                          (("SFA", 1, 0), ("SFB", 1, 0)),
                          config.AO.family("BPMx").count,
                          config.AO.family("HFC").count, config.AO.family("VFC").count)
