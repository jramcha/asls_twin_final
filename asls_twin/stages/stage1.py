#reviewed by jai ramchandani, 27/08/2026

"""Stage 1: mean of the previous three measured LOCO ORMs."""

from ..calculations import mean_orm #function which checks and takes mean of ORMs then returns this result.
from ._history import eligible_history


from .. import config

BACKGROUND_COUNT = config.HISTORY_COUNT
#this is the best performing lookback as of yet.


def calculate(state, history, *, workers=1, target_time=None, **context):
    del context
    del state, workers #again, state and workers unneeded
    records = eligible_history(history, target_time=target_time, require_ids=False)
    if len(records) < BACKGROUND_COUNT: # if we do not have enough history, throw an error. note for jai: consider changing this behaviour, adds a barrier to entry after a maintenance period.
        #currently, maintenance periods are not factored into this so will not be a problem.
        raise RuntimeError("Stage 1 requires three valid earlier LOCO measurements")
    return mean_orm([item.orm for item in records[-BACKGROUND_COUNT:]])#uses mean ORM to avg the LOCO orms.
