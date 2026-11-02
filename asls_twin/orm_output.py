"""Atomic forward-matrix export and trigger-driven exposure; no controller I/O."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.io import savemat

from . import config
from .systems import get_system
from .types import UpdateTracker
from .updates import advance_baseline, should_update


def export_orm(orm, path, *, metadata=None):
    """Atomically expose all four blocks plus their device and unit contract.

    Files contain the forward ORM, not an inverse or a controller wire format.
    A reader sees either the previous complete file or the new complete file.
    """
    orm.check()
    spec = get_system(orm.system)
    path = Path(path)
    if path.suffix not in (".npz", ".mat"):
        raise ValueError("ORM output must end in .npz or .mat")
    payload = {name: getattr(orm, name) for name in ("Rxx", "Ryx", "Rxy", "Ryy")}
    payload.update(system=orm.system, units=orm.units, contract=spec.fingerprint,
                   schema="forward-orm-v1", exported_at=datetime.now().isoformat(),
                   bpm_x_devices=np.array(config.AO.family("BPMx").devices),
                   bpm_y_devices=np.array(config.AO.family("BPMy").devices),
                   horizontal_devices=np.array(config.AO.family(spec.horizontal_family).devices),
                   vertical_devices=np.array(config.AO.family(spec.vertical_family).devices),
                   horizontal_family=spec.horizontal_family, vertical_family=spec.vertical_family,
                   metadata_json=json.dumps(metadata or {}, sort_keys=True, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if path.suffix == ".npz":
                np.savez_compressed(stream, **payload)
            else:
                savemat(stream, payload, do_compression=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


class ORMExposure:
    """One FOFB stage, trigger tracker and output file per instance.

    History discovery/training is separate. Failed exports leave the accepted
    trigger baseline and the previous complete matrix file intact.
    """

    def __init__(self, twin, stage, output, update, *, units=None, allow_experimental=False):
        self.twin, self.stage, self.output, self.update = twin, stage, Path(output), update
        self.units, self.allow_experimental = units, allow_experimental
        self.tracker = UpdateTracker()
        self.last_attempt = None
        update.selected_threshold

    def poll(self, state, history=None, *, beam_current_A=None, target_time=None, now_monotonic=None):
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.last_attempt is not None and now - self.last_attempt < 1 / self.update.max_push_hz:
            return False
        candidate = None
        def calculate():
            return self.twin.calculate(self.stage, state, history, beam_current_A=beam_current_A,
                                       target_time=target_time, units=self.units,
                                       allow_experimental=self.allow_experimental)
        if self.update.mode == "ormPredFrob":
            candidate = calculate()
            reference = self.tracker.accepted_orm
            if reference is None:
                due = True
            else:
                if candidate.system != reference.system or candidate.units != reference.units:
                    raise ValueError("ORM trigger requires matching system and units")
                denominator = np.linalg.norm(reference.upload_vector)
                if denominator <= 0:
                    raise ValueError("ORM trigger reference has zero norm")
                change = 100 * np.linalg.norm(candidate.upload_vector - reference.upload_vector) / denominator
                due = change >= self.update.selected_threshold
        else:
            due, _ = should_update(self.update, self.tracker, state, now_monotonic=now)
        if not due:
            return False
        self.last_attempt = now
        candidate = candidate if candidate is not None else calculate()
        export_orm(candidate, self.output, metadata=dict(
            stage=str(self.stage), id_state=dict(state), beam_current_A=beam_current_A,
            target_time=(target_time or datetime.now()).isoformat(),
            trigger=self.update.mode, scales=self.twin.scales, orm_method=self.twin.orm_method,
            stage6_contract=self.twin.stage6.context if str(self.stage) == "6h" else None,
            stage7_contract=self.twin.stage7.context if str(self.stage) == "7a" else None,
            training_record_keys=sorted(self.twin.stage7.keys if str(self.stage) == "7a"
                                        else self.twin.stage6.keys if str(self.stage) == "6h" else []),
            history=[dict(sha256=r.file_hash, timestamp=r.timestamp.isoformat())
                     for r in (history or [])]))
        advance_baseline(self.tracker, state, candidate, now_monotonic=now)
        return True
