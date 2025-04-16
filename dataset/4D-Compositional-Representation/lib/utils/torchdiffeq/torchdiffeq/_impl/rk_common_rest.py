# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import collections
from .misc import _scaled_dot_product, _convert_to_tensor

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha beta c_sol c_error')


class _RungeKuttaState(collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')):
    """Saved state of the Runge Kutta solver.

    Attributes:
        y1: Tensor giving the function value at the end of the last time step.
        f1: Tensor giving derivative at the end of the last time step.
        t0: scalar float64 Tensor giving start of the last time step.
        t1: scalar float64 Tensor giving end of the last time step.
        dt: scalar float64 Tensor giving the size for the next time step.
        interp_coef: list of Tensors giving coefficients for polynomial
            interpolation between `t0` and `t1`.
    """





