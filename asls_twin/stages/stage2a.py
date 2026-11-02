#reviewed by jai ramchandani version as of 27/08/2026

"""Stage 2a: ID-aware AT ORM with deployed Rxx/Ryy fudge factors."""

from ..calculations import at_to_ofb, id_aware_at_orm 
from ..safety import validate_id_state


def calculate(state, history, *, workers=1, **context):
    del context
    del history 
    return at_to_ofb(id_aware_at_orm(validate_id_state(state), workers=workers))#calculates the ID after using the validated ID state (checks for NaNs and inf and wrong shape), then 
    #calculates ID aware ORM optimally with monkeypatching and caching and removal of redundant functions, and then converts units and returns. 
