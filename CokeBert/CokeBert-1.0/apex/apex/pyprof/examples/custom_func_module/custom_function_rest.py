#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof
#Initialize pyprof
pyprof.nvtx.init()

class Foo(torch.autograd.Function):
	@staticmethod

	@staticmethod

#Hook the forward and backward functions to pyprof
pyprof.nvtx.wrap(Foo, 'forward')
pyprof.nvtx.wrap(Foo, 'backward')

foo = Foo.apply

x = torch.ones(4,4).cuda()
y = torch.ones(4,4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x,y)
	profiler.stop()