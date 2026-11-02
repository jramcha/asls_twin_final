#reviewed by jai ramchandani, 27/08/2026
"""Stage 3b: Stage 2a plus the mean of three measured-model residuals."""

from ..calculations import add_orm, mean_orm
from .stage2a import calculate as calculate_stage2
from ._history import eligible_history


# Tune this together with Stage 1b if the validated lookback changes. also consider the type of averaging being used, ref to note in stage 1b, check kalman and EWMA note for jai
from .. import config

BACKGROUND_COUNT = config.HISTORY_COUNT


def calculate(state, history, *, workers=1, target_time=None, **context):
    del context
    eligible = eligible_history(history, target_time=target_time, require_ids=True)
    if len(eligible) < BACKGROUND_COUNT: #if not enough lookback, throw error. Again note for jai, check compatibility for potential stage 3b if we are windowing after a maintenance period.
        #further, add this to the recalibration guide.
        raise RuntimeError(
            "Stage 3b requires three earlier LOCO measurements with ID-state records"
        )
    discrepancies = []
    for item in eligible[-BACKGROUND_COUNT:]:
        model = calculate_stage2(dict(item.id_state), [], workers=workers) #uses Stage 2a to calculate the ID-aware AT for each of the ID states in the lookback window.
        discrepancies.append(add_orm(item.orm, model, scale=-1.0)) # subtracts the ID aware ORMs from the LOCO matrices at each step, then adds each into an array (discrepancies). 
        #This is what I call "LOCO noise" or LOCO delta
    current = calculate_stage2(state, [], workers=workers)# calculates current ID aware ORM
    return add_orm(current, mean_orm(discrepancies)) #adds the mean loco noise from the last 3 noises to the current ID aware ORM. 
