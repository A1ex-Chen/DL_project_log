def original_mbconv(builder: LayerBuilder, depsep_kernel_size: int,
    in_channels: int, out_channels: int, expand_ratio: int, stride: int,
    squeeze_excitation_ratio: int, survival_prob: float, quantized: bool,
    trt: bool):
    return MBConvBlock(builder, depsep_kernel_size, in_channels,
        out_channels, expand_ratio, stride, squeeze_excitation_ratio,
        squeeze_hidden=False, survival_prob=survival_prob, quantized=
        quantized, trt=trt)
