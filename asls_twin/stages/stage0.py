"""Stage 0: stored SOFB baseline, retaining its original deployed scaling."""

import numpy as np
from scipy.io import loadmat

from .. import config
from ..types import ORM


def calculate(state, history, *, workers=1, **context):
    """Return the stored nm/A blocks unchanged; do not apply AT/LOCO scales.

    The reference contains only Rxx/Ryy. Cross-plane blocks are zero
    placeholders and are not part of the SOFB upload.
    """
    del state, history, workers, context
    reference = loadmat(config.DEPLOYED_ORM_REFERENCE_MAT)
    rxx = np.asarray(reference["Rxx_system"], dtype=float)
    ryy = np.asarray(reference["Ryy_system"], dtype=float)
    return ORM(rxx, np.zeros_like(rxx), np.zeros_like(ryy), ryy, "nm/A").check()
