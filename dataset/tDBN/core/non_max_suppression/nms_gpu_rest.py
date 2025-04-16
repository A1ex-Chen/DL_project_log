import math
from pathlib import Path

import numba
import numpy as np
from numba import cuda

from tDBN.utils.buildtools.pybind11_build import load_pb11

try:
    from tDBN.core.non_max_suppression.nms import non_max_suppression
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["../cc/nms/nms_kernel.cu.cc", "../cc/nms/nms.cc"],
        current_dir / "nms.so",
        current_dir,
        cuda=True)
    from tDBN.core.non_max_suppression.nms import non_max_suppression


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(int64, float32, float32[:, :], uint64[:])')


@cuda.jit('(int64, float32, float32[:], uint64[:])')


@numba.jit(nopython=True)


@numba.jit(nopython=True)






@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(float32[:], int32)', device=True, inline=True)


@cuda.jit('(float32[:], int32)', device=True, inline=True)


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)


@cuda.jit('(int64, float32, float32[:], uint64[:])')




@cuda.jit('(int64, int64, float32[:], float32[:], float32[:])', fastmath=False)




@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)


@cuda.jit(
    '(int64, int64, float32[:], float32[:], float32[:], int32)',
    fastmath=False)

