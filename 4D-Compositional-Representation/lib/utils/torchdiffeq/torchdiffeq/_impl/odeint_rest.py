from .tsit5 import Tsit5Solver
from .dopri5 import Dopri5Solver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .adams import VariableCoefficientAdamsBashforth
from .misc import _check_inputs

SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}

