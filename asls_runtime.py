"""Activate the checked-in Linux CPython 3.9 dependencies without installation."""
from pathlib import Path
import os
import platform
import sys

_ACTIVE = False

def activate():
    global _ACTIVE
    root = Path(__file__).resolve().parent
    if _ACTIVE:
        return root
    if os.environ.get("ASLS_TWIN_USE_SYSTEM_PACKAGES") == "1":
        _ACTIVE = True
        return root
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 9):
        raise RuntimeError("Bundled dependencies require CPython 3.9. Native development only: set ASLS_TWIN_USE_SYSTEM_PACKAGES=1.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Bundled dependencies require Linux x86_64. Native development only: set ASLS_TWIN_USE_SYSTEM_PACKAGES=1.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 17):
        raise RuntimeError("Bundled dependencies require glibc >= 2.17.")
    vendor = root / "vendor"
    for name in ("numpy", "scipy", "at", "epics", "tomli", "pyparsing"):
        module = sys.modules.get(name)
        if module is not None and vendor not in Path(getattr(module, "__file__", "")).resolve().parents:
            raise RuntimeError("Dependency already loaded outside vendor/: " + name + "; start a fresh Python process.")
    sys.path.insert(0, str(vendor))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYEPICS_LIBCA"] = str(vendor / "epics/clibs/linux64/libca.so")
    _ACTIVE = True
    return root
