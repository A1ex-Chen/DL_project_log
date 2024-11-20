def scale(self, wc, dc, dis, divisor=8) ->'EffNetArch':
    sw = EffNetArch._scale_width(wc, divisor=divisor)
    sd = EffNetArch._scale_depth(dc)
    return EffNetArch(block=self.block, stem_channels=sw(self.stem_channels
        ), feature_channels=sw(self.feature_channels), kernel=self.kernel,
        stride=self.stride, num_repeat=list(map(sd, self.num_repeat)),
        expansion=self.expansion, channels=list(map(sw, self.channels)),
        default_image_size=dis, squeeze_excitation_ratio=self.
        squeeze_excitation_ratio)
