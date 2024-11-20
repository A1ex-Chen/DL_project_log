import numba
from pathlib import Path
import numpy as np
from tDBN.utils.buildtools.pybind11_build import load_pb11

from tDBN.core.geometry import points_in_convex_polygon_3d_jit
from tDBN.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval

try:
    from tDBN.core import box_ops_cc
except:
    current_dir = Path(__file__).resolve().parents[0]
    box_ops_cc = load_pb11(["./cc/box_ops.cc"], current_dir / "box_ops_cc.so", current_dir)












@numba.njit

@numba.njit

@numba.njit


















@numba.jit(nopython=True)










































@numba.jit(nopython=True)




@numba.jit(nopython=False)


@numba.jit(nopython=True)




@numba.jit(nopython=True)


@numba.jit(nopython=True)


@numba.jit(nopython=True)


