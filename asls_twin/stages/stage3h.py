"""Stage 3h: Stage 2h plus the mean of three signed LOCO residuals."""

from ._history import residual_prediction
from .stage2h import calculate as calculate_stage2h


def calculate(
    state,
    history,
    *,
    workers=1,
    beam_current_A=None,
    target_time=None,
    machine_state=None,
    **context,
):
    del context
    return residual_prediction(
        state,
        history,
        baseline=calculate_stage2h,
        method="mean3",
        workers=workers,
        beam_current_A=beam_current_A,
        target_time=target_time,
        machine_state=machine_state,
    )
