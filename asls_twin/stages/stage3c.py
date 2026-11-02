"""Stage 3c: Stage 2a plus ID/current/time-weighted signed residuals."""

from ._history import residual_prediction
from .stage2a import calculate as calculate_stage2a


def calculate(
    state,
    history,
    *,
    workers=1,
    beam_current_A=None,
    target_time=None,
    **context,
):
    del context
    return residual_prediction(
        state,
        history,
        baseline=calculate_stage2a,
        method="ewma",
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
    )
