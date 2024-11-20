def get_model(self):
    vae_encoder = TorchVAEEncoder(self.model)
    return vae_encoder
