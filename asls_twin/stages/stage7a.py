#updated by jai ramchandani., 27/08/2026

"""Stage 7a: fit all four ORM blocks on eligible LOCO arrivals.

The service polls the LOCO directory and supplies validated history to update().
Offline calls without an active model retain the frozen reference behavior.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from .. import config
from ..loco_quality import MIN_HISTORY_CURRENT_A, accepted_history_current
from ..calculations import regression_features, regression_prediction
from ..safety import validate_id_state
from ..types import ORM
from ..systems import get_system


LOG = logging.getLogger(__name__)
VERSION = "active-stage7-v2-current-170ma"

# Identify a training record using its file contents, acquisition time and
# ordered ID settings. 
def record_key(record):
    payload = [record.file_hash, record.timestamp.isoformat(),
               [record.id_state[name] for name in config.ID_NAMES]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

# Keep a reusable fit for one feedback system. Training is done on history
class ActiveStage7:
    def __init__(self, path, training_start=None, *, system="SOFB", model_contract=None,
                 full_orm=True):
        self.system = system
        self.spec = get_system(system)
        self.full_orm = full_orm
        self.blocks = ("Rxx", "Ryx", "Rxy", "Ryy") if full_orm else ("Rxx", "Ryy")
        self.outputs = sum(int(np.prod(self.spec.shapes[name])) for name in self.blocks)
        self.path = Path(path) if path is not None else None
        self.training_start = training_start
        # The saved context ties the fit to the feature definition.
        context = [VERSION, MIN_HISTORY_CURRENT_A, str(training_start), config.REGRESSION_FEATURE_VERSION,
                   config.ID_NAMES, config.GAP_FEATURE_PERIOD_MM,
                   sorted(config.FIELD_SQUARED_IDS), config.ID_READBACK_RANGES,
                   config.FIELD_MAX_T, config.AO.sha256]
        if system != "SOFB" or model_contract is not None:
            context += [self.spec.fingerprint, model_contract]
        if full_orm:
            # Old diagonal fits must be rebuilt before supplying cross-plane values.
            context += ["full-orm-v1", self.blocks]
        self.context = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        self.model = None
        self.keys = set()
        # A missing or incompatible saved fit leaves the model unset. Training
        # can create a replacement when enough eligible measurements are available.
        if self.path is not None and self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as saved:
                    if str(saved["context"]) == self.context:
                        model = {key: saved[key].copy() for key in
                                 ("mean", "scale", "coef", "intercept", "low", "high")}
                        self._validate(model)
                        self.keys = set(saved["record_keys"].tolist())
                        self.model = model
            except Exception as exc:
                LOG.warning("Ignoring unusable Stage 7 model: %s", exc)

    # Check the feature count and the selected response-block layout.
    def _validate(self, model):
        p = len(config.ID_NAMES)
        outputs = self.outputs
        shapes = dict(mean=(p,), scale=(p,), coef=(p, outputs),
                      intercept=(outputs,), low=(p,), high=(p,))
        for key, shape in shapes.items():
            if model[key].shape != shape or not np.isfinite(model[key]).all():
                raise ValueError(f"invalid Stage 7 {key}")
        if np.any(model["scale"] <= 0) or np.any(model["low"] > model["high"]):
            raise ValueError("invalid Stage 7 scale/support")

    def update(self, history, *, now=None):
        """Fit all eligible records at or above 170 mA; never fit on prediction."""
        now = now or datetime.now()
        history = list(history)
        if any(record.orm.system != self.system for record in history):
            raise ValueError(f"Stage 7 {self.system} cannot train on another feedback system")
        records = {}
        hashes = set()
        # Filter before fitting: no future data, missing IDs, dates before the
        # training window or currents below the selected cutoff.
        for record in sorted(history, key=lambda r: (r.timestamp, str(r.path))):
            if (record.timestamp > now or record.id_state is None
                    or (self.training_start is not None and record.timestamp < self.training_start)):
                continue
            if not accepted_history_current(record.beam_current_A):
                LOG.info("Skipping LOCO record %s: beam current is missing or below 170 mA", record.path)
                continue
            try:
                validate_id_state(record.id_state)
                record.orm.check()
                if record.orm.units != "nm/A":
                    raise ValueError("Stage 7 requires nm/A")
            except (ValueError, TypeError) as exc:
                LOG.warning("Skipping Stage 7 record %s: %s", record.path, exc)
                continue
            if record.file_hash in hashes:
                continue
            hashes.add(record.file_hash)
            records[record_key(record)] = record
        keys = set(records)
        # Skip a refit when the accepted record identities have not changed.
        if keys == self.keys and self.model is not None:
            return False
        if len(records) < 2:
            LOG.warning("Stage 7 needs at least two distinct eligible LOCO records at or above 170 mA")
            self.model, self.keys = None, set()
            return False

        # Welford covariance avoids stacking an archive-sized target matrix.
        p = len(config.ID_NAMES)
        outputs = self.outputs
        mean = np.zeros(p)
        target_mean = np.zeros(outputs)
        xx = np.zeros((p, p))
        xy = np.zeros((p, outputs))
        low, high = np.full(p, np.inf), np.full(p, -np.inf)
        # Track raw readback bounds for the later support check.
        for n, record in enumerate(records.values(), 1):
            raw = np.array([record.id_state[name] for name in config.ID_NAMES])
            low, high = np.minimum(low, raw), np.maximum(high, raw)
            x = regression_features(record.id_state)
            y = np.concatenate([getattr(record.orm, name).ravel(order="C")
                                for name in self.blocks])
            dx, dy = x - mean, y - target_mean
            mean += dx / n
            target_mean += dy / n
            xx += np.outer(dx, x - mean)
            xy += np.outer(dx, y - target_mean)
        scale = np.sqrt(np.maximum(np.diag(xx) / n, 0))
        inactive = scale < 1e-9
        scale[inactive] = 1.
        covariance = xx / np.outer(scale, scale)
        cross = xy / scale[:, None]
        covariance[inactive, :] = 0
        covariance[:, inactive] = 0
        cross[inactive] = 0
        coef = np.linalg.solve(covariance + np.eye(p), cross)  # Fixed ridge alpha=1.
        model = dict(mean=mean, scale=scale, coef=coef, intercept=target_mean,
                     low=low, high=high)
        self._validate(model)
        if self.path is not None:
            if self.path.exists():
                try:
                    with np.load(self.path, allow_pickle=False) as previous:
                        previous_system = str(previous["system"]) if "system" in previous else "SOFB"
                except Exception as exc:
                    # Keep the broken artifact for inspection, then rebuild.
                    backup = self.path.with_name(self.path.name + ".corrupt")
                    if backup.exists():
                        raise RuntimeError(f"cannot quarantine corrupt Stage 7 model: {backup} exists") from exc
                    os.replace(self.path, backup)
                    LOG.warning("Quarantined unusable Stage 7 model at %s: %s", backup, exc)
                else:
                    if previous_system != self.system:
                        raise ValueError(f"refusing to overwrite a {previous_system} model with {self.system}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write the complete candidate fit to a temporary file, then replace the
            # previous file.
            temporary = self.path.with_suffix(".tmp.npz")
            try:
                np.savez_compressed(temporary, **model, context=self.context,
                                    version=VERSION, system=self.system, record_keys=sorted(keys),
                                    blocks=self.blocks,
                                    training_start=str(self.training_start), alpha=1.,
                                    fitted_at=now.isoformat(),
                                    provenance=json.dumps([
                                        dict(path=str(r.path), sha256=r.file_hash,
                                             timestamp=r.timestamp.isoformat(), id_state=dict(r.id_state),
                                             beam_current_A=r.beam_current_A)
                                        for r in records.values()]))
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        # A failed fit/save leaves the preceding in-memory model intact.
        self.model, self.keys = model, keys
        return True

    # Predict within listed ID limits; warn when outside observed training coverage.
    def predict(self, state):
        if self.model is None:
            raise RuntimeError("Stage 7 has no eligible active fit; waiting for valid LOCO records")
        state = validate_id_state(state) # Raises ValueError if outside of hard limits
        raw = np.array([state[name] for name in config.ID_NAMES])
        model = self.model

        # Allow extrapolation within listed limits. We do not need a ValueError here, and can instead log a warning
        outside_training = (raw < model["low"]) | (raw > model["high"])
        if outside_training.any():
            LOG.warning(
                "Stage 7 extrapolating beyond observed training bounds for: %s",
                ", ".join(np.array(config.ID_NAMES)[outside_training]),
            )
        vector = model["intercept"] + ((regression_features(state) - model["mean"])
                                        / model["scale"]) @ model["coef"]
        values = {name: np.zeros(shape) for name, shape in self.spec.shapes.items()}
        offset = 0
        for name in self.blocks:
            shape = self.spec.shapes[name]
            size = int(np.prod(shape))
            values[name] = vector[offset:offset + size].reshape(shape)
            offset += size
        return ORM(values["Rxx"], values["Ryx"], values["Rxy"], values["Ryy"],
                   "nm/A", self.system).check()

# Use the service's active fit when one is supplied. Offline calls without
# an active fit retain the packaged frozen Stage 7 regression.
def calculate(state, history, *, workers=1, **context):
    active = context.get("stage7_model")
    if active is not None:
        return active.predict(state)
    del context
    del history, workers
    return regression_prediction(state, "stage7") #uses the regression function (Stage 7a setting) to use the frozen regression table which it imports to predict the
    #ORM based on the ID state alone
