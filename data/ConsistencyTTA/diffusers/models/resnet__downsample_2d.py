def _downsample_2d(self, hidden_states, weight=None, kernel=None, factor=2,
    gain=1):
    """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight:
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] *
            factor`, which corresponds to average pooling.
            factor: Integer downsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and
            same datatype as `x`.
        """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * gain
    if self.use_conv:
        _, _, convH, convW = weight.shape
        pad_value = kernel.shape[0] - factor + (convW - 1)
        stride_value = [factor, factor]
        upfirdn_input = upfirdn2d_native(hidden_states, torch.tensor(kernel,
            device=hidden_states.device), pad=((pad_value + 1) // 2, 
            pad_value // 2))
        output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0
            )
    else:
        pad_value = kernel.shape[0] - factor
        output = upfirdn2d_native(hidden_states, torch.tensor(kernel,
            device=hidden_states.device), down=factor, pad=((pad_value + 1) //
            2, pad_value // 2))
    return output
