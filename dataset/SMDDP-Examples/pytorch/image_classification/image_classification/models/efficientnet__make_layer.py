def _make_layer(self, block, kernel_size, stride, num_repeat, expansion,
    in_channels, out_channels, squeeze_excitation_ratio, prev_layer_count, trt
    ):
    layers = []
    idx = 0
    survival_prob = self._get_survival_prob(idx + prev_layer_count)
    blk = block(self.builder, kernel_size, in_channels, out_channels,
        expansion, stride, self.arch.squeeze_excitation_ratio, 
        survival_prob if stride == 1 and in_channels == out_channels else 
        1.0, self.quantized, trt=trt)
    layers.append((f'block{idx}', blk))
    for idx in range(1, num_repeat):
        survival_prob = self._get_survival_prob(idx + prev_layer_count)
        blk = block(self.builder, kernel_size, out_channels, out_channels,
            expansion, 1, squeeze_excitation_ratio, survival_prob, self.
            quantized, trt=trt)
        layers.append((f'block{idx}', blk))
    return nn.Sequential(OrderedDict(layers)), out_channels
