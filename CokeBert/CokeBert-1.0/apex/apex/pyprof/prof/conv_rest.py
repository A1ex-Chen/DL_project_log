from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Conv(OperatorLayerBase):

	"""
	# N = batch size
	# C,H,W = input channels, height, width
	# K,P,Q = output channels, height, width
	# R,S = filter height, width
	# g = groups
	"""

	#todo: refine winograd and FFT
	convAuxList = ["nchwToNhwc", "nhwcToNchw", "OffsetsKernel",]
	winoAuxList = ["generateWinogradTilesKernel", "winogradWgradData", "winogradWgradOutput", "winogradWgradDelta"]
	fftAuxList = ["compute_gemm_pointers", "flip_filter", "fft2d_r2c_", "fft2d_c2r_", "fft1d_r2c", "fft1d_c2r"]
	miscAuxList = ["scaleTensor_kernel",]

	convList = ["_s884cudnn_", "_scudnn_", "2d_grouped_direct_kernel", "cudnn::detail::implicit_convolve_sgemm", "cudnn::detail::dgrad2d_alg1_1", "cudnn::detail::wgrad_alg0_engine", "cudnn::detail::dgrad_engine", "dgrad_1x1_stride_2x2", "spatialDepthwiseConvolutionUpdateOutput"]
	winoList = ["winograd3x3Kernel", "_sgemm_"]
	fftList = ["fermiPlusCgemmLDS128_batched", "_gcgemm_",]
	miscList = []








