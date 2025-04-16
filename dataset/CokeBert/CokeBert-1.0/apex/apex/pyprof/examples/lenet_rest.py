#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.optim as optim

from apex import pyprof
pyprof.nvtx.init()

class LeNet5(nn.Module):



with torch.autograd.profiler.emit_nvtx():

	net = LeNet5().cuda()

	input = torch.randn(1, 1, 32, 32).cuda()
	out = net(input)

	target = torch.randn(10)			# a dummy target, for example
	target = target.view(1, -1).cuda()	# make it the same shape as output
	criterion = nn.MSELoss()

	# create your optimizer
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	# in your training loop:
	optimizer.zero_grad()	# zero the gradient buffers

	profiler.start()
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()	# Does the update
	profiler.stop()
