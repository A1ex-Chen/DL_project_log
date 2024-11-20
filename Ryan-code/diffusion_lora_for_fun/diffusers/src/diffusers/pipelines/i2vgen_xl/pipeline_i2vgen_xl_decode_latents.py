def decode_latents(self, latents, decode_chunk_size=None):
    latents = 1 / self.vae.config.scaling_factor * latents
    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size *
        num_frames, channels, height, width)
    if decode_chunk_size is not None:
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            frame = self.vae.decode(latents[i:i + decode_chunk_size]).sample
            frames.append(frame)
        image = torch.cat(frames, dim=0)
    else:
        image = self.vae.decode(latents).sample
    decode_shape = (batch_size, num_frames, -1) + image.shape[2:]
    video = image[None, :].reshape(decode_shape).permute(0, 2, 1, 3, 4)
    video = video.float()
    return video
