"""File-based FOFB training and matrix exposure. Never connects to EPICS."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np

from .config import UpdateConfig
from .fofb import FOFBTwin, STAGES
from .loco import Stage7HistoryReader
from .orm_output import ORMExposure, export_orm
from .safety import validate_id_state


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="3h_weighted")
    parser.add_argument("--state", type=Path, help="JSON with id_state, beam_current_A and timestamp")
    parser.add_argument("--history", type=Path, action="append",
                        help="FOFB ORM history root; repeat for ordinary and LHS results")
    parser.add_argument("--registry", type=Path, default=Path("state/fofb/registry.json"))
    parser.add_argument("--output", type=Path, help="Atomic forward ORM export (.npz or .mat)")
    parser.add_argument("--scales", type=float, nargs=2, metavar=("HORIZONTAL", "VERTICAL"))
    parser.add_argument("--units", choices=("m/rad", "nm/A"))
    parser.add_argument("--orm-method", choices=("linear", "bipolar"), default="linear")
    parser.add_argument("--stage7-model", type=Path, default=Path("state/fofb/stage7_active.npz"))
    parser.add_argument("--stage6-model", type=Path, default=Path("state/fofb/stage6_residual.npz"))
    parser.add_argument("--training-start", type=datetime.fromisoformat)
    parser.add_argument("--train-stage7", action="store_true")
    parser.add_argument("--train-stage6", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Watch state JSON and expose on configured trigger")
    parser.add_argument("--mode", choices=("time", "idFrob", "idFrobCubed", "ormPredFrob"), default="time")
    parser.add_argument("--period", type=float, default=60)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--poll", type=float, default=2)
    parser.add_argument("--history-poll", type=float, default=600)
    parser.add_argument("--state-max-age", type=float, default=60)
    parser.add_argument("--max-export-hz", type=float, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.fit_only and not (args.train_stage6 or args.train_stage7):
        parser.error("--fit-only requires --train-stage6 or --train-stage7")
    if args.watch and (args.train_stage6 or args.fit_only):
        parser.error("Stage 6 training is an offline operation; use a separate invocation")
    if not args.fit_only and (args.output is None or (args.state is None and args.stage != "0")):
        parser.error("matrix generation requires --output and --state (except Stage 0)")
    if args.watch and args.state is None:
        parser.error("--watch requires --state with a refreshed timestamp")
    for name in ("poll", "history_poll", "state_max_age", "period", "threshold", "max_export_hz"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.output is not None:
        reserved = [args.state, args.registry, args.stage6_model, args.stage7_model]
        if any(path is not None and args.output.resolve() == path.resolve() for path in reserved):
            parser.error("--output must be separate from state, registry and model files")
    twin = FOFBTwin(scales=args.scales, stage6_model_file=args.stage6_model,
                    stage7_model_file=args.stage7_model, training_start=args.training_start,
                    orm_method=args.orm_method)
    reader = (Stage7HistoryReader(args.history, args.registry, args.training_start,
                                 grace_seconds=30 if args.watch else 0, system="FOFB")
              if args.history is not None else None)
    history = reader.read() if reader is not None else []
    if args.train_stage6:
        twin.fit_stage6(history)
        if twin.stage6.model is None:
            raise RuntimeError("insufficient eligible FOFB records to fit Stage 6")
    if args.train_stage7 or (args.watch and args.stage == "7a"):
        twin.update_stage7(history)
        if not args.watch and twin.stage7.model is None:
            raise RuntimeError("insufficient eligible FOFB records to fit Stage 7")
    if args.fit_only:
        print(json.dumps(dict(stage6_fitted=twin.stage6.model is not None,
                              stage7_fitted=twin.stage7.model is not None)))
        return 0
    update = UpdateConfig(args.mode, args.poll, args.period, args.threshold,
                          args.threshold, args.threshold, args.max_export_hz)
    exposure = ORMExposure(twin, args.stage, args.output, update, units=args.units,
                           allow_experimental=args.allow_experimental)
    next_history = time.monotonic() + args.history_poll
    while True:
        try:
            now = datetime.now()
            if args.watch and reader is not None and time.monotonic() >= next_history:
                next_history = time.monotonic() + args.history_poll
                history = reader.read(now=now)
                if args.stage == "7a":
                    twin.update_stage7(history, now=now)
            raw = json.loads(args.state.read_text()) if args.state is not None else {}
            state = validate_id_state(raw["id_state"]) if args.stage != "0" or args.watch else {}
            timestamp = datetime.fromisoformat(raw["timestamp"]) if "timestamp" in raw else now
            if args.watch and ("timestamp" not in raw or not 0 <= (now - timestamp).total_seconds() <= args.state_max_age):
                raise ValueError("state JSON timestamp is missing, future or stale")
            current = raw.get("beam_current_A")
            if not args.watch:
                orm = twin.calculate(args.stage, state, history, beam_current_A=current,
                                      target_time=timestamp, units=args.units,
                                      allow_experimental=args.allow_experimental)
                export_orm(orm, args.output, metadata=dict(stage=args.stage, id_state=state,
                    beam_current_A=current, target_time=timestamp.isoformat(), scales=twin.scales,
                    orm_method=twin.orm_method,
                    stage6_contract=twin.stage6.context, stage7_contract=twin.stage7.context,
                    training_record_keys=sorted(twin.stage7.keys if args.stage == "7a"
                                               else twin.stage6.keys if args.stage == "6h" else []),
                    history=[dict(sha256=r.file_hash, timestamp=r.timestamp.isoformat())
                             for r in history]))
                print(json.dumps(dict(output=str(args.output.resolve()), system=orm.system,
                                      units=orm.units, Rxx=list(orm.Rxx.shape), Ryy=list(orm.Ryy.shape))))
                return 0
            if exposure.poll(state, history, beam_current_A=current, target_time=timestamp):
                logging.info("Exposed FOFB Stage %s matrix: %s", args.stage, args.output)
        except Exception:
            if not args.watch:
                raise
            logging.exception("FOFB exposure skipped; keeping the previous complete output")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(cli())
