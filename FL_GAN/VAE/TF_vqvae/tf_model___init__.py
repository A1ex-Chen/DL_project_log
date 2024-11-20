def __init__(self, color_channels, pooling_kernels, decoder_input_size):
    super().__init__()
    self.decoder_input_size = decoder_input_size
    self.m1 = DecoderModule(256, 128, stride=1)
    self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
    self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
    self.bottle = DecoderModule(32, color_channels, stride=1, activation='tanh'
        )
