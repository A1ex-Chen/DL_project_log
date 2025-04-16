import numpy as np
from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Pointwise(OperatorLayerBase):

	ops = []
	ops += ["__abs__", "__neg__", "__invert__"]
	ops += ["__add__", "__sub__", "__mul__", "__floordiv__", "__truediv__", "__pow__", "__mod__"]
	ops += ["__radd__", "__rsub__", "__rmul__", "__rdiv__", "__rtruediv__", "__rfloordiv__", "__rpow__"]
	ops += ["__iadd__", "__isub__", "__imul__", "__itruediv__",]
	ops += ["__lt__", "__gt__", "__ge__", "__le__", "__eq__", "__ne__",]
	ops += ["lt", "lt_", "gt", "gt_", "ge", "ge_", "le", "le_", "eq", "eq_", "ne", "ne_",]
	ops += ["__and__", "__or__", "__xor__", "__lshift__", "__rshift__"]
	ops += ["__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]
	ops += ["abs", "abs_", "neg", "neg_"]
	ops += ["add", "add_", "div", "div_", "mul", "mul_", "reciprocal", "reciprocal_", "remainder", "remainder_", "sub", "sub_",]
	ops += ["addcdiv", "addcdiv_", "addcmul", "addcmul_"]
	ops += ["exp", "exp_", "exp1m", "exp1m_", "log", "log_", "log10", "log10_", "log1p", "log1p_", "log2", "log2_", "pow", "pow_", "rsqrt", "rsqrt_", "sqrt", "sqrt_",]
	ops += ["ceil", "ceil_", "clamp", "clamp_", "floor", "floor_", "fmod", "fmod_", "frac", "frac_", "round", "round_", "sign", "sign_", "trunc", "trunc_"]
	ops += ["acos", "acos_", "asin", "asin_", "atan", "atan_", "atan2", "atan2_", "cos", "cos_", "cosh", "cosh_", "sin", "sin_", "sinh", "sinh_", "tan", "tan_", "sigmoid", "sigmoid_", "tanh", "tanh_"]
	ops += ["digamma", "erf", "erf_", "erfc", "erfc_", "erfinv", "erfinv_", "lerp", "lerp_", "mvlgamma",]

	@staticmethod







