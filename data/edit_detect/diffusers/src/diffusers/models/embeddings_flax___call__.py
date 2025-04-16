@nn.compact
def __call__(self, timesteps):
    return get_sinusoidal_embeddings(timesteps, embedding_dim=self.dim,
        flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift)
