def forward(self, x):
    return retrieve_latents(self.vae_encoder.encode(x))
