def decode_latents(self, latents):
    latents = 1 / self.vae.config.scaling_factor * latents
    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size *
        num_frames, channels, height, width)
    image = self.vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.
        shape[2:]).permute(0, 2, 1, 3, 4)
    video = video.float()
    return video
