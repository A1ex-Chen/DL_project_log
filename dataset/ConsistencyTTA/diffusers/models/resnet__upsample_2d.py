def _upsample_2d(self, hidden_states, weight=None, kernel=None, factor=2,
    gain=1):
    """Fused `upsample_2d()` followed by `Conv2d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight: Weight tensor of the shape `[filterH, filterW, inChannels,
                outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
                (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
            factor: Integer upsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
            datatype as `hidden_states`.
        """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * (gain * factor ** 2)
    if self.use_conv:
        convH = weight.shape[2]
        convW = weight.shape[3]
        inC = weight.shape[1]
        pad_value = kernel.shape[0] - factor - (convW - 1)
        stride = factor, factor
        output_shape = (hidden_states.shape[2] - 1) * factor + convH, (
            hidden_states.shape[3] - 1) * factor + convW
        output_padding = output_shape[0] - (hidden_states.shape[2] - 1
            ) * stride[0] - convH, output_shape[1] - (hidden_states.shape[3
            ] - 1) * stride[1] - convW
        assert output_padding[0] >= 0 and output_padding[1] >= 0
        num_groups = hidden_states.shape[1] // inC
        weight = torch.reshape(weight, (num_groups, -1, inC, convH, convW))
        weight = torch.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
        weight = torch.reshape(weight, (num_groups * inC, -1, convH, convW))
        inverse_conv = F.conv_transpose2d(hidden_states, weight, stride=
            stride, output_padding=output_padding, padding=0)
        output = upfirdn2d_native(inverse_conv, torch.tensor(kernel, device
            =inverse_conv.device), pad=((pad_value + 1) // 2 + factor - 1, 
            pad_value // 2 + 1))
    else:
        pad_value = kernel.shape[0] - factor
        output = upfirdn2d_native(hidden_states, torch.tensor(kernel,
            device=hidden_states.device), up=factor, pad=((pad_value + 1) //
            2 + factor - 1, pad_value // 2))
    return output
