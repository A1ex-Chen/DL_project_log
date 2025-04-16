from copy import deepcopy
from itertools import repeat
from functools import partial

from deeplite_torch_zoo.src.zero_cost_proxies.fisher import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.grad_norm import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.grasp import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.jacob_cov import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.l2_norm import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.plain import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.snip import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.synflow import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.zico.zico import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.nwot.nwot import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.nparams import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.macs import *  # pylint: disable=unused-import

from deeplite_torch_zoo.utils import weight_gaussian_init
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES





    return compute_zc_score_wrapper