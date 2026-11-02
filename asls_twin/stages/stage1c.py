"""Stage 1c: ID/current/time-weighted historical LOCO ORM."""

from ._history import eligible_history, history_weights, weighted_orm


def calculate(
    state,
    history,
    *,
    beam_current_A=None,
    target_time=None,
    **context,
):
    del context
    records = eligible_history(history, target_time=target_time, require_ids=True)
    weights = history_weights(records, state, beam_current_A, target_time)
    return weighted_orm([item.orm for item in records], weights)
