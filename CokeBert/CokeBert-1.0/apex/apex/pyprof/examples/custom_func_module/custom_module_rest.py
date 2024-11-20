#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof
pyprof.nvtx.init()

class Foo(torch.nn.Module):


#Hook the forward function to pyprof
pyprof.nvtx.wrap(Foo, 'forward')

foo = Foo(4)
foo.cuda()
x = torch.ones(4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x)
	profiler.stop()