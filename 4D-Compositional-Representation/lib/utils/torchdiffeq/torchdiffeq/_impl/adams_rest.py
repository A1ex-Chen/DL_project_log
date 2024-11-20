import collections
import torch
from .solvers import AdaptiveStepsizeODESolver
from .misc import (
    _handle_unused_kwargs, _select_initial_step, _convert_to_tensor, _scaled_dot_product, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)

_MIN_ORDER = 1
_MAX_ORDER = 12

gamma_star = [
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
]


class _VCABMState(collections.namedtuple('_VCABMState', 'y_n, prev_f, prev_t, next_t, phi, order')):
    """Saved state of the variable step size Adams-Bashforth-Moulton solver as described in

        Solving Ordinary Differential Equations I - Nonstiff Problems III.5
        by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett.
    """






class VariableCoefficientAdamsBashforth(AdaptiveStepsizeODESolver):



