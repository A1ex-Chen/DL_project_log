def __init__(self, d):
    self.name = d.name
    self.dir = d.dir
    self.sub = d.sub
    marker = eval(d.argMarker[0])
    mod = marker['mod']
    op = marker['op']
    args = marker['args']
    assert mod == 'torch.nn.functional'
    assert op == 'linear'
    self.setXWBMNK(args)
    if any(x in d.name for x in Linear.gemmKernels):
        self.op_ = 'linear'
    else:
        assert d.name in Linear.biasKernels
        self.op_ = 'bias'
    """
		elif (("kernelPointwiseApply2" in d.name) or ("kernelReduceContigDim" in d.name) or ("kernelReduceNoncontigDim_shared" in d.name)):
			#bias expansion was before the gemm
			self.op_ = "bias"

		elif ("elementwise_kernel" in d.name):
			#Bias addition happens later with a broadcast tensor
			self.op_ = "bias"
			assert (len(d.argMarker) == 2)
			marker = eval(d.argMarker[1])
			mod = marker['mod']
			op = marker['op']
			args = marker['args']

			assert (mod == "Tensor")
			assert (op == "__iadd__")
			assert (len(args) == 2)
			mn = args[0]['shape']
			b = args[1]['shape']
			assert (len(b) == 1)

			assert (mn == (self.n + (self.m,)))
			assert (b == self.b)

		else:
			assert False
		"""
