"""Stage 3a: Stage 2a plus the last measured-minus-physics residual."""

from ._history import residual_prediction
from .stage2a import calculate as calculate_stage2a


def calculate(state, history, *, workers=1, target_time=None, **context):
    return residual_prediction(
        state,
        history,
        baseline=calculate_stage2a,
        method="last",
        workers=workers,
        beam_current_A=None,
        target_time=target_time,
    )
