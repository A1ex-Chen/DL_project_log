def forward(self, x):
    return self.vae_encoder.encode(x).latent_dist.sample()
