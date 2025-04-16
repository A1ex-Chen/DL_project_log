"""
This file intercepts (monkey patches) the following functions and adds NVTX markers.
	torch.*
	torch.Tensor.*
	torch.nn.functional.*
	torch.nn.*.forward

The NVTX markers (one or more) contain the following information
	call trace (a list of file_name:line_number)
	extra_repr() from torch.nn modules
	module/class name
	function name
	inputs (args and kwargs)
		scalar: name, type and value
		tensor: name, shape and datatype
		numpy: name, shape and datatype
		list/tuple: a sequence of scalars or tensors or numpy arrays
"""

import torch
import torch.cuda.nvtx as nvtx
import numpy
import inspect as ins
import traceback
import math







	setattr(mod, fn_name, wrapper_func)

def argMarker(mod, op, args, kwargs):
	#For this function args is a tuple and kwargs is a dict








	cadena = {}
	cadena['mod'] = mod.__name__
	cadena['op'] = op
	cadena['args'] = []

	foo(args, "")
	for k,v in kwargs.items():
		foo((v,), k)

	return str(cadena)

def patchClass(cls):
	for f in dir(cls):
		if isfunc(cls, f):
			add_wrapper(cls, f)

def init():
	print("Initializing NVTX monkey patches")
	for cls in [torch, torch.Tensor, torch.nn.functional,]:
		patchClass(cls)

	for cls in [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.GRUCell]:
		if isfunc(cls, 'forward'):
			add_wrapper(cls, 'forward')

	print("Done with NVTX monkey patching")