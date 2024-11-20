import math
from pathlib import Path

from tDBN.utils.buildtools.pybind11_build import load_pb11
from tDBN.utils.find import find_cuda_device_arch
import numba
import numpy as np

try:
    from tDBN.core.non_max_suppression.nms import (
        non_max_suppression_cpu, rotate_non_max_suppression_cpu)
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["../cc/nms/nms_kernel.cu.cc", "../cc/nms/nms.cc"],
        current_dir / "nms.so",
        current_dir,
        cuda=True)
    from tDBN.core.non_max_suppression.nms import (
        non_max_suppression_cpu, rotate_non_max_suppression_cpu)

from tDBN.core import box_np_ops






@numba.jit(nopython=True)


@numba.jit('float32[:, :], float32, float32, float32, uint32', nopython=True)