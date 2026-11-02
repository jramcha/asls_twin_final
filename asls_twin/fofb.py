"""Offline FOFB stages, training and bounded per-instance prediction caches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from . import calculations, config, orm_generation
from .stages.stage7a import ActiveStage7
from .safety import validate_id_state
from .stages._history import eligible_history, history_weights
from .stages._stage2h_physics import build_stage2h_ring
from .systems import get_system
from .types import ORM


BLOCKS = ("Rxx", "Ryx", "Rxy", "Ryy")
STAGES = ("0", "1", "2a", "2h", "3h", "3h_weighted", "6h", "7a")


def combine(values, weights):
    if not values:
        raise ValueError("cannot combine an empty FOFB history")
    weights = np.asarray(weights, float)
    if weights.shape != (len(values),) or not np.isfinite(weights).all():
        raise ValueError("invalid FOFB combination weights")
    units = values[0].units
    for value in values:
        value.check()
        if value.system != "FOFB" or value.units != units:
            raise ValueError("FOFB combination requires matching systems and units")
    return ORM(*(sum((weight * getattr(value, block)
                      for value, weight in zip(values, weights)),
                     np.zeros_like(getattr(values[0], block))) for block in BLOCKS),
               units, "FOFB").check()


def _check_history(history):
    seen = {}
    for record in history:
        record.orm.check()
        if record.orm.system != "FOFB" or record.orm.units != "nm/A":
            raise ValueError("FOFB history must contain FOFB measurements in nm/A")
        signature = (record.timestamp, record.id_state, record.beam_current_A)
        if record.file_hash in seen and seen[record.file_hash] != signature:
            raise ValueError("conflicting metadata for duplicate FOFB measurement")
        seen[record.file_hash] = signature


def _contract(scales, *, residual, orm_method="linear"):
    digest = hashlib.sha256()
    digest.update(get_system("FOFB").fingerprint.encode())
    digest.update(json.dumps([config.REGRESSION_FEATURE_VERSION, config.ID_NAMES,
                              config.GAP_FEATURE_PERIOD_MM,
                              sorted(config.FIELD_SQUARED_IDS)], sort_keys=True).encode())
    files = ["calculations.py", "safety.py", "systems.py", "types.py"]
    if residual:
        digest.update(json.dumps(scales).encode())
        digest.update(orm_method.encode())
        files += ["orm_generation.py", "fofb.py", "stages/_stage2h_physics.py",
                  "stages/_history.py"]
        for path in (config.MODEL_REFERENCE_FILE, config.LATTICE_MAT,
                     config.M66_FITS_MAT, config.QUAD_FF_MAT):
            digest.update(Path(path).read_bytes())
    for name in files:
        digest.update((config.PACKAGE / name).read_bytes())
    return digest.hexdigest()


class FOFBTwin:
    """Expose matrices only. No EPICS reads, writes, or implicit SOFB artifacts.

    Stage 6 is trained against causal Stage 3h-weighted predictions. Stage 7
    fits all four measured blocks directly. Both use fixed ridge alpha=1.
    """

    def __init__(self, *, scales=None, stage7_model_file=None, stage6_model_file=None,
                 training_start=None, cache_size=128, orm_method="linear"):
        if orm_method not in ("linear", "bipolar"):
            raise ValueError("ORM method must be linear or bipolar")
        self._orm_method = orm_method
        if scales is not None:
            factors = np.asarray(scales, float)
            if factors.shape != (2,) or not np.isfinite(factors).all() or np.any(factors <= 0):
                raise ValueError("FOFB scales must be two positive finite factors")
            scales = tuple(float(value) for value in factors)
        if not isinstance(cache_size, int) or cache_size < 1 or cache_size > 512:
            raise ValueError("cache_size must be in 1..512")
        if stage6_model_file is not None and stage7_model_file is not None:
            if Path(stage6_model_file).resolve() == Path(stage7_model_file).resolve():
                raise ValueError("Stage 6 and Stage 7 need separate model files")
        self._scales, self.cache_size = scales, cache_size
        self.training_start = training_start
        self._cache = OrderedDict()
        self.stage7 = ActiveStage7(stage7_model_file, training_start, system="FOFB",
                                  model_contract=_contract(None, residual=False))
        self.stage6_contract = _contract(scales, residual=True, orm_method=orm_method)
        self.stage6 = ActiveStage7(stage6_model_file, training_start, system="FOFB",
                                  model_contract=self.stage6_contract, full_orm=False)

    @property
    def scales(self):
        return self._scales

    @property
    def orm_method(self):
        return self._orm_method

    def _physics(self, stage, state, current, *, workers=1):
        state = validate_id_state(state)
        key = (stage, tuple(state[name] for name in config.ID_NAMES), current)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        if stage == "2a":
            raw = orm_generation.id_aware_at_orm(state, system="FOFB", workers=workers,
                                                method=self.orm_method)
        else:
            ring, indices = build_stage2h_ring(state, current)
            raw = orm_generation.build_orm(ring, system="FOFB", workers=workers, indices=indices,
                                           method=self.orm_method)
        self._cache[key] = raw.copy()
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return raw

    def _scaled_physics(self, stage, state, current, *, workers=1):
        if self.scales is None:
            raise ValueError("FOFB Stages 3h/6h require FOFB scales; no SOFB factors are reused")
        return orm_generation.at_to_ofb(self._physics(stage, state, current, workers=workers),
                                         scales=self.scales)

    def _stage3(self, stage, state, history, current, target, *, workers=1):
        records = eligible_history(history, target_time=target, require_ids=True)
        method = {"3h": "mean3", "3h_weighted": "weighted"}[stage]
        count = {"mean3": 3}.get(method)
        if not records or (count is not None and len(records) < count):
            raise RuntimeError("insufficient earlier ID-associated FOFB measurements")
        records = records[-count:] if count is not None else records
        weights = (history_weights(records, state, current, target) if method == "weighted"
                   else np.ones(len(records)) / len(records))
        physics = "2h"
        residuals = [combine([r.orm, self._scaled_physics(
            physics, dict(r.id_state), r.beam_current_A, workers=workers)], [1, -1])
                     for r in records]
        baseline = self._scaled_physics(physics, state, current, workers=workers)
        return combine([baseline, combine(residuals, weights)], [1, 1])

    def calculate(self, stage, state, history=None, *, beam_current_A=None,
                  target_time=None, workers=1, units=None, allow_experimental=False):
        stage = str(stage).lower()
        if stage not in STAGES:
            raise ValueError(f"FOFB stage {stage!r} is unavailable; choose from {STAGES}")
        if workers != 1:
            raise ValueError("FOFB generation currently requires workers=1")
        history = list(history or [])
        _check_history(history)
        # Copies of one measurement must not count as independent history.
        unique = {}
        for record in history:
            if record.file_hash in unique:
                prior = unique[record.file_hash]
                if (prior.id_state != record.id_state or prior.timestamp != record.timestamp
                        or prior.beam_current_A != record.beam_current_A):
                    raise ValueError("conflicting metadata for duplicate FOFB measurement")
            unique[record.file_hash] = record
        history = sorted(unique.values(), key=lambda r: r.timestamp)
        target = target_time or datetime.now()
        if stage in ("0", "2a", "2h"):
            wanted = units or ("nm/A" if self.scales is not None else "m/rad")
            if wanted not in ("m/rad", "nm/A"):
                raise ValueError("FOFB physics output units must be m/rad or nm/A")
            if wanted == "nm/A" and self.scales is None:
                raise ValueError("FOFB nm/A output requires explicit FOFB scales")
            result = (orm_generation.static_at_orm(system="FOFB", workers=workers, method=self.orm_method)
                      if stage == "0" else self._physics(stage, state, beam_current_A, workers=workers))
            if wanted == "nm/A":
                return orm_generation.at_to_ofb(result, scales=self.scales)
            return result
        if units not in (None, "nm/A"):
            raise ValueError("measured and learned FOFB stages return nm/A")
        if stage == "1":
            records = eligible_history(history, target_time=target, require_ids=False)
            count = config.HISTORY_COUNT
            if len(records) < count:
                raise RuntimeError("insufficient earlier FOFB measurements")
            records = records[-count:]
            return combine([r.orm for r in records], np.ones(count) / count)
        if stage == "7a":
            return self.stage7.predict(state)
        if stage == "6h":
            correction = self.stage6.predict(state)
            baseline = self._stage3("3h_weighted", state, history, beam_current_A, target)
            return combine([baseline, correction], [1, 1])
        return self._stage3(stage, state, history, beam_current_A, target, workers=workers)

    def update_stage7(self, history, *, now=None):
        history = list(history)
        _check_history(history)
        return self.stage7.update(history, now=now)

    def fit_stage6(self, history, *, now=None):
        """Train residuals from predict-before-update targets; never self-fit a target."""
        history = list(history)
        _check_history(history)
        if self.scales is None:
            raise ValueError("Stage 6 training requires FOFB scales")
        now = now or datetime.now()
        records = eligible_history(history, target_time=now, require_ids=True)
        records = [r for r in records if self.training_start is None or r.timestamp >= self.training_start]
        unique = {}
        for record in records:
            if record.file_hash in unique:
                prior = unique[record.file_hash]
                if prior.timestamp != record.timestamp or prior.id_state != record.id_state:
                    raise ValueError("conflicting duplicate FOFB measurement")
            unique[record.file_hash] = record
        records = sorted(unique.values(), key=lambda r: r.timestamp)
        residual_records = []
        for record in records:
            earlier = [r for r in records if r.timestamp < record.timestamp]
            if not earlier:
                continue
            predicted = self._stage3("3h_weighted", dict(record.id_state), earlier,
                                     record.beam_current_A, record.timestamp)
            residual = combine([record.orm, predicted], [1, -1])
            # Bind the correction target to all prior records and their metadata.
            cohort = [(r.file_hash, r.timestamp.isoformat(), dict(r.id_state), r.beam_current_A)
                      for r in earlier + [record]]
            digest = hashlib.sha256((self.stage6_contract + json.dumps(cohort, sort_keys=True)).encode()).hexdigest()
            residual_records.append(replace(record, orm=residual, file_hash=digest))
        return self.stage6.update(residual_records, now=now)
