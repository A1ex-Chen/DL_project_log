#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof

class Foo(torch.jit.ScriptModule):

    @torch.jit.script_method

#Initialize pyprof after the JIT step
pyprof.nvtx.init()

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(Foo, 'forward')

foo = Foo(4)
foo.cuda()
x = torch.ones(4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x)
	profiler.stop()
	print(z)