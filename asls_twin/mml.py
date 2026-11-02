"""Validated hardware metadata from a MATLAB Middle Layer AO export.

MATLAB function handles and saved AT indices are metadata only. We do not
execute conversion callbacks or apply indices from a different lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Family:
    name: str
    devices: tuple[tuple[int, int], ...]
    members: tuple[str, ...]
    channels: tuple[str, ...]
    hw_units: str
    physics_units: str
    monitor: Mapping

    @property
    def count(self):
        return len(self.devices)

    def channel(self, device):
        try:
            return self.channels[self.devices.index(tuple(device))]
        except ValueError as exc:
            raise ValueError(f'AO.{self.name}: missing device {device}') from exc


def _strings(value):
    array = np.asarray(value)
    # loadmat normally joins MATLAB char rows; also accept explicit char arrays.
    if array.ndim == 2 and array.dtype.kind in 'US' and array.dtype.itemsize <= 4:
        return tuple(''.join(row).strip() for row in array.tolist())
    return tuple(str(x).strip() for x in array.ravel())


class MMLAO:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        data = loadmat(BytesIO(payload), simplify_cells=True)
        roots = [data[k] for k in ('ao', 'AO') if k in data]
        if len(roots) != 1 or not isinstance(roots[0], dict):
            raise ValueError('AO file must contain exactly one ao or AO structure')
        self.raw = roots[0]
        self.families = {}
        for name, value in self.raw.items():
            if not isinstance(value, dict) or 'DeviceList' not in value:
                continue
            devices = np.asarray(value['DeviceList'], dtype=float)
            if devices.ndim == 1 and devices.shape == (2,):
                devices = devices.reshape(1, 2)
            if (devices.ndim != 2 or devices.shape[1] != 2 or not len(devices)
                    or not np.isfinite(devices).all() or np.any(devices < 1)
                    or not np.array_equal(devices, np.floor(devices))):
                raise ValueError(f'AO.{name}: invalid DeviceList')
            rows = tuple(tuple(int(x) for x in row) for row in devices)
            if len(set(rows)) != len(rows):
                raise ValueError(f'AO.{name}: duplicate DeviceList entries')
            monitor = value.get('Monitor', {})
            channels = _strings(monitor.get('ChannelNames', []))
            self.families[name] = Family(name, rows, _strings(value.get('MemberOf', [])),
                                         channels, str(monitor.get('HWUnits', '')).strip(),
                                         str(monitor.get('PhysicsUnits', '')).strip(), monitor)

    def family(self, name):
        try:
            return self.families[name]
        except KeyError as exc:
            raise ValueError(f'AO is missing required family {name}') from exc

    def members(self, tag):
        return tuple(k for k, v in self.families.items() if tag in v.members)

    def validate(self, reference):
        """The packaged AO fixes the model/ORM row contract; PV names may change."""
        names = ('BPMx', 'BPMy', 'HCM', 'VCM', 'ID', 'DCCT') + reference.members('QUAD') + reference.members('SEXT')
        for name in names:
            actual, expected = self.family(name), reference.family(name)
            if actual.devices != expected.devices:
                raise ValueError(f'AO.{name} DeviceList differs from the packaged model ordering')
            if actual.hw_units != expected.hw_units or actual.physics_units != expected.physics_units:
                raise ValueError(f'AO.{name} units differ from the supported model units')
            if (len(actual.channels) != actual.count or not all(actual.channels)
                    or len(set(actual.channels)) != actual.count):
                raise ValueError(f'AO.{name} has missing or duplicate monitor channels')
        for tag in ('QUAD', 'SEXT'):
            if set(self.members(tag)) != set(reference.members(tag)):
                raise ValueError(f'AO {tag} family membership differs from the model')
        if self.family('BPMx').devices != self.family('BPMy').devices:
            raise ValueError('AO BPMx/BPMy device ordering differs')
        # These values enter a fixed nm/A ORM and mm ID-gap contract. Changed
        # conversions must not be silently interpreted using the previous units.
        for name in ('BPMx', 'BPMy', 'ID'):
            if not np.array_equal(self.linear_scale(name), reference.linear_scale(name)):
                raise ValueError(f'AO.{name} conversion differs from the model unit contract')
        self.linear_scale('DCCT')
        return self

    def linear_scale(self, family):
        value = self.family(family)
        monitor = value.monitor
        if 'HW2PhysicsFcn' in monitor:
            raise ValueError(f'AO.{family} uses a MATLAB conversion function, not a linear scale')
        scale = np.asarray(monitor.get('HW2PhysicsParams'), dtype=float)
        if scale.size == 1:
            scale = np.full(value.count, float(scale.ravel()[0]))
        if scale.shape != (value.count,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f'AO.{family} has invalid HW2PhysicsParams')
        offset = np.asarray(monitor.get('Offset', 0), dtype=float)
        if not np.isfinite(offset).all() or np.any(offset != 0):
            raise ValueError(f'AO.{family} nonzero monitor offsets require an explicit conversion')
        return scale

    def calibration_curve(self, family):
        """Common amp2k table for the existing scalar diagnostic, not deployment."""
        value = self.family(family)
        table = np.asarray(value.monitor.get('HW2PhysicsParams'), dtype=float)
        if (table.ndim != 2 or table.shape[0] != value.count or table.shape[1] % 2
                or table.shape[1] < 4 or not np.isfinite(table).all()):
            raise ValueError(f'AO.{family}: malformed amp2k calibration table')
        if not np.all(table == table[0]):
            raise ValueError(f'AO.{family}: per-device calibration cannot be represented by one diagnostic slope')
        curve = table[0].copy()
        axis = curve[:len(curve) // 2]
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f'AO.{family}: calibration axis must increase')
        return curve
