import math
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import importlib

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

class FusedLayerNormAffineFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    bias_ = bias.contiguous()
    output, mean, invvar = fused_layer_norm_cuda.forward_affine(
        input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
    ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)
    return grad_input, grad_weight, grad_bias, None, None

class FusedLayerNormFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, normalized_shape, eps):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    output, mean, invvar = fused_layer_norm_cuda.forward(
        input_, ctx.normalized_shape, ctx.eps)
    ctx.save_for_backward(input_, mean, invvar)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input_, mean, invvar = ctx.saved_tensors
    grad_input = None
    grad_input = fused_layer_norm_cuda.backward(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        ctx.eps)
    return grad_input, None, None



  @staticmethod

class FusedLayerNormFunction(torch.autograd.Function):

  @staticmethod

  @staticmethod

def fused_layer_norm_affine(input, normalized_shape, weight, bias, eps=1e-6):
    return FusedLayerNormAffineFunction.apply(input, weight, bias, normalized_shape, eps)

def fused_layer_norm(input, normalized_shape, eps=1e-6):
    return FusedLayerNormFunction.apply(input, normalized_shape, eps)

class FusedLayerNorm(torch.nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedLayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedLayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """


