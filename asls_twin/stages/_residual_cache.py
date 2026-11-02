"""Disk cache of measured-minus-model ORM residuals.

A residual is fixed once its measurement and its baseline model are fixed, so it
is worth computing at most once. Without this cache every weighted Stage 3
evaluation rebuilds one AT model per historical measurement, on every poll.

Entries are keyed by a fingerprint of everything that determines the baseline:
the packaged physics data, the OFB scale factors that carry m/rad to nm/A, and
the baseline source itself. If any of those change the whole store is discarded
rather than silently mixing residuals taken against different models.

Storage has two modes. "exact" keeps each residual as a float32 vector and is
lossless. "lowrank" keeps a shared mean and basis with per-entry amplitudes; it
is much smaller for a long archive but it is LOSSY, so compact() is explicit,
never automatic, and reports the reconstruction error it costs.
"""

from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..types import ORM


BLOCKS = ("Rxx", "Ryx", "Rxy", "Ryy")
VERSION = "1"

_STORE_FILE: Path | None = None
_STORE: "ResidualStore | None" = None


def configure(path: str | Path | None) -> None:
    """Point the cache at a file, or disable it with None."""
    global _STORE_FILE, _STORE
    _STORE_FILE = Path(path).expanduser().resolve() if path else None
    _STORE = None


def store() -> "ResidualStore | None":
    global _STORE
    if _STORE_FILE is None:
        return None
    if _STORE is None:
        _STORE = ResidualStore(_STORE_FILE)
    return _STORE


_DATA_FILES = (
    config.MODEL_REFERENCE_FILE,
    config.LATTICE_MAT,
    config.QUAD_FF_MAT,
    config.M66_FITS_MAT,
)


def source_fingerprint_bytes(path: Path) -> bytes:
    """Fingerprint parsed source consistently on Python 3.9 and 3.12.

    Tokenization of f-strings changed in Python 3.12. The parsed expression
    tree preserves their executable content without depending on those tokens
    or source locations. Empty type_params fields were also added in 3.12;
    omit only that empty default, retaining any actual type parameters.
    """
    def canonical(value):
        if isinstance(value, ast.AST):
            return (type(value).__name__, tuple(
                (name, canonical(field)) for name, field in ast.iter_fields(value)
                if not (name == "type_params" and field == [])
            ))
        if isinstance(value, list):
            return tuple(canonical(item) for item in value)
        return value

    return repr(canonical(ast.parse(path.read_bytes()))).encode("utf-8")


@lru_cache(maxsize=None)
def baseline_fingerprint(module_name: str) -> str:
    """Hash the data, scale factors and source that determine a baseline ORM."""
    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update(b"python-ast-runtime-v3")
    digest.update(f"{config.OFB_RXX_SCALE!r}|{config.OFB_RYY_SCALE!r}".encode())
    for path in _DATA_FILES:
        digest.update(Path(path).read_bytes())
    sources = [Path(__file__).parent / f"{module_name.rsplit('.', 1)[-1]}.py"]
    # Only executable baseline dependencies belong in the physics fingerprint.
    sources += [Path(__file__).parent / name
                for name in ("_stage2h_physics.py", "_current_focusing.py")]
    sources += [
        config.PACKAGE / name
        for name in ("calculations.py", "safety.py", "mml.py")
    ]
    for path in sources:
        digest.update(source_fingerprint_bytes(path))
    return digest.hexdigest()


def entry_key(module_name: str, item) -> str:
    """Identify one residual: which baseline, which measurement, which state."""
    state = dict(item.id_state or {})
    state_text = ";".join(f"{name}={state[name]!r}" for name in sorted(state))
    current = "none" if item.beam_current_A is None else f"{item.beam_current_A!r}"
    detail = hashlib.sha256(f"{state_text}|{current}".encode()).hexdigest()[:16]
    return f"{module_name}|{baseline_fingerprint(module_name)}|{item.file_hash}|{detail}"


def flatten(orm: ORM) -> np.ndarray:
    orm.check()
    if orm.units != "nm/A":
        raise ValueError("residual cache stores nm/A only")
    return np.concatenate([np.asarray(getattr(orm, b), float).ravel() for b in BLOCKS])


def unflatten(vector: np.ndarray) -> ORM:
    shapes = {
        "Rxx": (config.N_BPM, config.N_HCM),
        "Ryx": (config.N_BPM, config.N_HCM),
        "Rxy": (config.N_BPM, config.N_VCM),
        "Ryy": (config.N_BPM, config.N_VCM),
    }
    blocks = {}
    offset = 0
    for name in BLOCKS:
        rows, columns = shapes[name]
        size = rows * columns
        blocks[name] = np.asarray(
            vector[offset : offset + size], float
        ).reshape(rows, columns)
        offset += size
    if offset != len(vector):
        raise ValueError(f"residual vector has {len(vector)} entries, expected {offset}")
    return ORM(*(blocks[name] for name in BLOCKS), "nm/A").check()


class ResidualStore:
    """Keyed residual vectors on disk, exact by default."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.keys: list[str] = []
        self.mode = "exact"
        self.vectors: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.amplitudes: np.ndarray | None = None
        self.remainder: np.ndarray | None = None
        self._dirty = False
        self._load()

    def _reset(self) -> None:
        self.keys = []
        self.mode = "exact"
        self.vectors = None
        self.mean = self.basis = self.amplitudes = self.remainder = None

    def _load(self) -> None:
        if not self.path.exists():
            self._reset()
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                if str(data["version"]) != VERSION:
                    raise ValueError("residual cache version mismatch")
                self.keys = [str(k) for k in data["keys"]]
                self.mode = str(data["mode"])
                if self.mode == "exact":
                    self.vectors = data["vectors"]
                else:
                    self.mean = data["mean"]
                    self.basis = data["basis"]
                    self.amplitudes = data["amplitudes"]
                    self.remainder = data["remainder"]
        except Exception:
            self._reset()

    def _index(self, key: str) -> int | None:
        try:
            return self.keys.index(key)
        except ValueError:
            return None

    def get(self, key: str) -> ORM | None:
        position = self._index(key)
        if position is None:
            return None
        if self.mode == "exact":
            return unflatten(self.vectors[position])
        return unflatten(self.mean + self.amplitudes[position] @ self.basis)

    def put(self, key: str, residual: ORM) -> None:
        if self._index(key) is not None:
            return
        vector = flatten(residual)
        if self.mode == "exact":
            row = vector.astype(np.float32)[None, :]
            self.vectors = row if self.vectors is None else np.vstack((self.vectors, row))
        else:
            centered = vector - self.mean
            amplitude = self.basis @ centered
            leftover = float(np.linalg.norm(centered - amplitude @ self.basis))
            self.amplitudes = np.vstack(
                (self.amplitudes, amplitude.astype(np.float32)[None, :])
            )
            self.remainder = np.concatenate(
                (self.remainder, np.float32([leftover]))
            )
        self.keys.append(key)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        payload = {
            "version": np.str_(VERSION),
            "mode": np.str_(self.mode),
            "keys": np.asarray(self.keys, dtype=np.str_),
        }
        if self.mode == "exact":
            payload["vectors"] = self.vectors
        else:
            payload.update(
                mean=self.mean,
                basis=self.basis,
                amplitudes=self.amplitudes,
                remainder=self.remainder,
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(temporary, **payload)
        os.replace(temporary, self.path)
        self._dirty = False

    def compact(self, rank: int) -> dict[str, float]:
        """Replace exact vectors with a rank-truncated basis. LOSSY.

        Returns the captured energy fraction and the reconstruction error in
        nm/A, so the caller can compare it against the estimator it feeds.
        """
        if self.mode != "exact":
            raise RuntimeError("residual cache is already compacted")
        if self.vectors is None or len(self.keys) < 2:
            raise RuntimeError("residual cache needs at least two entries to compact")
        if rank < 1 or rank > len(self.keys):
            raise ValueError(f"rank must lie in 1..{len(self.keys)}")
        vectors = np.asarray(self.vectors, float)
        mean = vectors.mean(axis=0)
        centered = vectors - mean
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        basis = right[:rank]
        amplitudes = centered @ basis.T
        error = centered - amplitudes @ basis
        total = float((singular**2).sum())
        energy = float((singular[:rank] ** 2).sum() / total) if total > 0 else 1.0
        self.mode = "lowrank"
        self.mean = mean
        self.basis = basis.astype(np.float32)
        self.amplitudes = amplitudes.astype(np.float32)
        self.remainder = np.linalg.norm(error, axis=1).astype(np.float32)
        self.vectors = None
        self._dirty = True
        return {
            "rank": float(rank),
            "entries": float(len(self.keys)),
            "captured_energy": energy,
            "rms_error_nm_per_A": float(np.sqrt((error**2).mean())),
            "max_abs_error_nm_per_A": float(np.abs(error).max()),
        }

    def report(self) -> dict[str, float]:
        stored = 0
        if self.mode == "exact" and self.vectors is not None:
            stored = self.vectors.nbytes
        elif self.basis is not None:
            stored = self.basis.nbytes + self.amplitudes.nbytes + self.mean.nbytes
        return {
            "entries": float(len(self.keys)),
            "bytes": float(stored),
            "mean_remainder": (
                float(np.mean(self.remainder)) if self.remainder is not None else 0.0
            ),
        }
