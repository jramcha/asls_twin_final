"""Stage 2h: ID-aware physics with open-gap Ky rebasing and current shift."""

from __future__ import annotations

from ..calculations import at_to_ofb, build_orm
from ._stage2h_physics import build_stage2h_ring


def calculate(
    state,
    history,
    *,
    workers=1,
    beam_current_A=None,
    machine_state=None,
    **context,
):
    """Build the Ky-rebased, current-aware baseline without a measured residual."""
    del history, context
    if machine_state is not None:
        raise ValueError("Stage 2h has no committed extended machine-state calibration")
    ring, indices = build_stage2h_ring(state, beam_current_A)
    return at_to_ofb(build_orm(ring, workers=workers, indices=indices))
