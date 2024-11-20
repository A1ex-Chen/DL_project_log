def __init__(self, in_channels, squeeze, activation, quantized=False,
    use_conv=False):
    super().__init__(in_channels, squeeze, activation, use_conv=use_conv)
    self.quantized = quantized
    if quantized:
        assert quant_nn is not None, 'pytorch_quantization is not available'
        self.mul_a_quantizer = quant_nn.TensorQuantizer(quant_nn.
            QuantConv2d.default_quant_desc_input)
        self.mul_b_quantizer = quant_nn.TensorQuantizer(quant_nn.
            QuantConv2d.default_quant_desc_input)
