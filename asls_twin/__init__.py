# last updated by Jai Ramchandani, 25/08/2026
# reviewed by jai ramchandani, version as of 27/08/2026

"""Production API for the Australian Synchrotron digital twin."""

#base file, used when import asls_twin is called (e.g. in stageN.py). 

#__all__ defines which functions are standalone functions (or classes and datatypes) which are exported when somebody calls from asls_twin import * , and it is essentially all of the functions which would
# ever perceptibly be used outside of the normal asls_twin scope.

from asls_runtime import activate as _activate_runtime
_activate_runtime()

from .stages import calculate_stage, stage_catalog #changed to import stages within the library
from .epics import read_id_state, write_orm
from .loco import read_latest_loco, read_loco_history
from .safety import choose_safe_orm
from .types import LocoMeasurement, ORM, SafetyDecision, WriteResult
from .orm_generation import build_orm, static_at_orm, id_aware_at_orm
from .fofb import FOFBTwin

__version__ = "2.0.0"

__all__ = [
    "LocoMeasurement",
    "FOFBTwin",
    "build_orm",
    "static_at_orm",
    "id_aware_at_orm",
    "ORM",
    "SafetyDecision",
    "WriteResult",
    "calculate_stage",
    "stage_catalog",
    "choose_safe_orm",
    "read_id_state",
    "read_latest_loco",
    "read_loco_history",
    "write_orm",
]
