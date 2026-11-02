#!/usr/bin/env python3
# Last reviewed: 20/09/2026
"""Long-running ASLS digital-twin service."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

from asls_twin import calculate_stage, choose_safe_orm, read_id_state, write_orm
from asls_twin.config import RuntimeConfig, load_config
from asls_twin import config as machine_config
from asls_twin.epics import (UnsafeMachineStateError, load_verified_reference,
                             read_beam_current, read_current_orm)
from asls_twin.loco import read_loco_history, Stage7HistoryReader
from asls_twin.stages.stage7a import ActiveStage7
from asls_twin.safety import normalized_id_strengths
from asls_twin.stages import CURRENT_STAGES
from asls_twin.stages import _residual_cache
from asls_twin.types import ORM, UpdateTracker
from asls_twin.updates import advance_baseline, should_update


LOG = logging.getLogger("asls_twin")
_STOP = False


def _stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _event(cfg: RuntimeConfig, kind: str, **fields: object) -> None:
    record = {"timestamp": datetime.now().isoformat(), "event": kind, **fields}
    cfg.paths.event_log.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.event_log.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_reference(cfg: RuntimeConfig, orm: ORM, stage: str) -> None:
    target = cfg.paths.reference_file
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.npz")
    orm.save(temporary)
    os.replace(temporary, target)
    metadata = {
        "verified": True,
        "written_at": datetime.now().isoformat(),
        "stage": stage,
        "sha256": _sha256(target),
    }
    meta_tmp = cfg.paths.reference_metadata_file.with_suffix(".tmp")
    meta_tmp.parent.mkdir(parents=True, exist_ok=True)
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(meta_tmp, cfg.paths.reference_metadata_file)


def _persisted_reference(cfg: RuntimeConfig) -> ORM | None:
    meta = cfg.paths.reference_metadata_file
    if not meta.exists() or not cfg.paths.reference_file.exists():
        return None
    try:
        value = json.loads(meta.read_text())
        if value.get("verified") is not True:
            return None
        if value.get("sha256") != _sha256(cfg.paths.reference_file):
            return None
    except (OSError, json.JSONDecodeError):
        return None
    return load_verified_reference(cfg.paths.reference_file)


# Review synthesis (Nick): prefer the matrix currently in use; only a
# verified persisted reference is eligible as the startup fallback.
def _startup_reference(cfg: RuntimeConfig) -> tuple[ORM | None, str]:
    try:
        return read_current_orm(), "live in-use OFB matrix"
    except Exception as exc:
        LOG.warning("cannot read in-use OFB matrix: %s", exc)
    persisted = _persisted_reference(cfg)
    if persisted is not None:
        return persisted, "verified persisted reference"
    return None, "no usable reference"


# Review synthesis (Nick): the service reads machine state on its main
# cadence, but discovers LOCO history only on loco_poll_seconds. A Stage 7
# refit updates the model; the selected update policy still controls pushes.
def run(cfg: RuntimeConfig, *, enable_writes: bool, once: bool = False) -> int:
    ao = machine_config.configure_mml(cfg.ao_mml_file)
    _residual_cache.configure(cfg.paths.residual_cache_file)
    _event(cfg, "machine_configuration", ao_mml_file=str(ao.path), ao_sha256=ao.sha256,
           model_reference_file=str(machine_config.MODEL_REFERENCE_FILE),
           model_reference_sha256=_sha256(machine_config.MODEL_REFERENCE_FILE))
    reference, reference_source = _startup_reference(cfg)
    armed = cfg.writes_enabled and enable_writes and reference is not None
    LOG.info(
        "stage=%s update=%s reference=%s writes=%s",
        cfg.stage,
        cfg.update.mode,
        reference_source,
        "ARMED" if armed else "dry-run",
    )
    if reference is None:
        LOG.error("no current or verified reference; the service will remain dry-run")

    stage7_model = (ActiveStage7(cfg.stage7_model_file, cfg.stage7.training_start)
                    if cfg.stage == "7a" else None)
    stage7_reader = (Stage7HistoryReader(
        cfg.stage7.history_directories or (cfg.paths.loco_directory,),
        cfg.paths.registry_file, cfg.stage7.training_start,
        cfg.loco_complete_grace_seconds) if stage7_model is not None else None)
    history = []
    history_error = None
    next_loco_poll = float("-inf")
    tracker = UpdateTracker()
    last_push_started: float | None = None
    while not _STOP:
        started = time.monotonic()
        rejected_candidate = False
        try:
            state = read_id_state()
            beam_current_A = None
            if cfg.stage in CURRENT_STAGES:
                beam_current_A = (
                    read_beam_current() * machine_config.BEAM_CURRENT_HW_TO_A
                )
            if started >= next_loco_poll:
                # Schedule before I/O: failures must not turn into four-second retries.
                next_loco_poll = started + cfg.loco_poll_seconds
                try:
                    if stage7_reader is not None:
                        history = stage7_reader.read()
                        if not history:
                            raise RuntimeError("Stage 7 is waiting for eligible LOCO records")
                        age = (datetime.now() - history[-1].timestamp).total_seconds()
                        if age < 0 or age > cfg.loco_max_age_seconds:
                            raise RuntimeError("newest eligible LOCO measurement is stale or future-dated")
                        if stage7_model.update(history):
                            _event(cfg, "stage7_refit", training_count=len(stage7_model.keys),
                                   training_start=str(cfg.stage7.training_start),
                                   model_file=str(cfg.stage7_model_file))
                    else:
                        history = read_loco_history(
                            cfg.paths.loco_directory,
                            registry_file=cfg.paths.registry_file,
                            max_age_seconds=cfg.loco_max_age_seconds,
                            complete_grace_seconds=cfg.loco_complete_grace_seconds,
                            capture_window_seconds=cfg.loco_capture_window_seconds,
                            capture_new=True,
                        )
                    history_error = None
                except Exception as exc:
                    history_error = exc
                    raise
            if history_error is not None:
                raise RuntimeError(f"LOCO refresh failed; waiting for next discovery: {history_error}")
            # Freshness still expires between discovery polls.
            if history:
                age = (datetime.now() - history[-1].timestamp).total_seconds()
                if age < 0 or age > cfg.loco_max_age_seconds:
                    raise RuntimeError("newest eligible LOCO measurement is stale or future-dated")

            candidate = None
            if cfg.update.mode == "ormPredFrob" or reference is None:
                candidate = calculate_stage(
                    cfg.stage,
                    state,
                    history,
                    workers=cfg.workers,
                    beam_current_A=beam_current_A,
                    target_time=datetime.now(),
                    stage7_model=stage7_model,
                )
            trigger, trigger_value = should_update(
                cfg.update,
                tracker,
                state,
                now_monotonic=started,
                candidate=candidate,
                reference=reference,
            )
            if reference is None:
                trigger = True
            push_ready = (
                last_push_started is None
                or started - last_push_started >= 1.0 / cfg.update.max_push_hz
            )
            if trigger and push_ready:
                candidate = candidate or calculate_stage(
                    cfg.stage,
                    state,
                    history,
                    workers=cfg.workers,
                    beam_current_A=beam_current_A,
                    target_time=datetime.now(),
                    stage7_model=stage7_model,
                )
                decision = choose_safe_orm(
                    candidate,
                    reference,
                    max_candidate_change_percent=cfg.safety.max_candidate_change_percent,
                )
                _event(
                    cfg,
                    "candidate",
                    stage=cfg.stage,
                    update_mode=cfg.update.mode,
                    trigger_value=trigger_value,
                    accepted=decision.accepted,
                    change_percent=decision.change_percent,
                    reason=decision.reason,
                )
                if not decision.accepted:
                    rejected_candidate = True
                    LOG.warning("candidate rejected: %s", decision.reason)
                else:
                    if (cfg.writes_enabled and enable_writes
                            and reference is not None and decision.should_write):
                        # Prediction can take many seconds. A changed input
                        # invalidates it; the writer also checks the live ORM.
                        fresh_state = read_id_state()
                        for name in machine_config.ID_NAMES:
                            kind = machine_config.ID_PVS[name][1]
                            tolerance = machine_config.LIVE_ID_CHANGE_TOLERANCE[kind]
                            if abs(fresh_state[name] - state[name]) > tolerance:
                                raise UnsafeMachineStateError(
                                    f"{name} changed during prediction; rearm after checking machine state")
                        if beam_current_A is not None:
                            fresh_current_A = (read_beam_current()
                                * machine_config.BEAM_CURRENT_HW_TO_A)
                            if abs(fresh_current_A - beam_current_A) > 0.020:
                                raise UnsafeMachineStateError(
                                    "beam current changed during prediction; rearm after checking machine state")
                        # Count failed attempts too, using actual dispatch time.
                        last_push_started = time.monotonic()
                    result = write_orm(
                        decision,
                        writes_enabled=cfg.writes_enabled,
                        enable_writes=enable_writes and reference is not None,
                        readback_rtol=cfg.safety.readback_rtol,
                        readback_atol_nm_per_a=cfg.safety.readback_atol_nm_per_a,
                        expected_reference=reference if cfg.writes_enabled and enable_writes else None,
                        max_candidate_change_percent=cfg.safety.max_candidate_change_percent,
                    )
                    _event(cfg, "write", **result.__dict__)
                    if result.verified:
                        reference = candidate
                        _persist_reference(cfg, candidate, cfg.stage)
                        advance_baseline(tracker, state, candidate, now_monotonic=started)
                        LOG.info("candidate written, verified, and persisted")
                    elif not result.attempted:
                        advance_baseline(tracker, state, candidate, now_monotonic=started)
                        LOG.info("validated dry run: %s", result.reason)
                    else:
                        LOG.error("write did not verify: %s", result.reason)
        except UnsafeMachineStateError as exc:
            LOG.critical("live writes disarmed; verify machine state and explicitly rearm: %s", exc)
            _event(cfg, "unsafe_machine_state", error=repr(exc), writes_disarmed=True)
            return 2
        except Exception as exc:
            LOG.exception("poll failed: %s", exc)
            _event(cfg, "poll_error", error=repr(exc))
            if once:
                return 1

        if once:
            return 1 if rejected_candidate else 0
        remaining = cfg.update.poll_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        else:
            LOG.warning(
                "poll cycle took %.2fs, longer than poll_seconds=%.2fs; "
                "running back-to-back",
                time.monotonic() - started,
                cfg.update.poll_seconds,
            )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="second write arm; config must also set writes_enabled=true",
    )
    parser.add_argument("--once", action="store_true", help="run one polling cycle")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level_name = args.log_level.upper()
    level = getattr(logging, level_name, None)
    if level is None:
        level = logging.INFO
        print(
            f"Unknown log level '{args.log_level}', defaulting to INFO",
            file=sys.stderr,
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    return run(load_config(args.config), enable_writes=args.enable_writes, once=args.once)


if __name__ == "__main__":
    sys.exit(cli())
