def __init__(self, K=128, D=256, channel=3):
    super().__init__()
    self.codebook = VectorQuantizer(K=K, D=D)
    self.encoder = Encoder(D=D, in_channel=channel)
    self.decoder = Decoder(D=D, out_channel=channel)
