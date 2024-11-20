def __init__(self, K, D, **kwargs):
    super().__init__(**kwargs)
    self.K = K
    self.D = D
    self.codebook = VectorQuantizer(self.K, self.D)
    self.encoder = Encoder(D=self.D)
    self.decoder = Decoder(D=self.D)
