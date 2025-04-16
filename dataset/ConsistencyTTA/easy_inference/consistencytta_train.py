def train(self, mode: bool=True):
    self.unet.train(mode)
    for model in [self.text_encoder, self.vae, self.fn_STFT]:
        model.eval()
    return self
