def decode_latents(self, latents):
    latents = 1 / self.vae.config.scaling_factor * latents
    mel_spectrogram = self.vae.decode(latents).sample
    return mel_spectrogram
