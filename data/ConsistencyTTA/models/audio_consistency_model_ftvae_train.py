def train(self, mode: bool=True):
    super().train(mode)
    self.ema_vae_decoder.eval()
    self.vae.decoder.train(mode)
    return self
