from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Activation(OperatorLayerBase):
	"""
	This class handles the various activation functions.
	"""

	ops = ["celu", "elu", "elu_", "hardshrink", "hardtanh", "hardtanh_", "leaky_relu", "leaky_relu_", "logsigmoid", "prelu", "relu", "relu_", "relu6", "rrelu", "rrelu_", "selu", "sigmoid", "softplus", "softshrink", "softsign", "tanh", "tanhshrink", "threshold", "threshold_"]






