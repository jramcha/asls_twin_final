"""Stage 1a: the last valid measured LOCO ORM."""

from ._history import eligible_history


def calculate(state, history, *, target_time=None, **context):
    del state, context
    records = eligible_history(history, target_time=target_time, require_ids=False)
    if not records:
        raise RuntimeError("Stage 1a requires an earlier LOCO measurement")
    return records[-1].orm.copy()
