def decode_latents(self, latents: torch.Tensor, num_frames: int,
    decode_chunk_size: int=14):
    latents = latents.flatten(0, 1)
    latents = 1 / self.vae.config.scaling_factor * latents
    forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae
        ) else self.vae.forward
    accepts_num_frames = 'num_frames' in set(inspect.signature(
        forward_vae_fn).parameters.keys())
    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        num_frames_in = latents[i:i + decode_chunk_size].shape[0]
        decode_kwargs = {}
        if accepts_num_frames:
            decode_kwargs['num_frames'] = num_frames_in
        frame = self.vae.decode(latents[i:i + decode_chunk_size], **
            decode_kwargs).sample
        frames.append(frame)
    frames = torch.cat(frames, dim=0)
    frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2,
        1, 3, 4)
    frames = frames.float()
    return frames
