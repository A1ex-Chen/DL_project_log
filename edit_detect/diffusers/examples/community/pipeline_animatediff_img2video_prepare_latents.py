def prepare_latents(self, image, strength, batch_size, num_channels_latents,
    num_frames, height, width, dtype, device, generator, latents=None,
    latent_interpolation_method='slerp'):
    shape = (batch_size, num_channels_latents, num_frames, height // self.
        vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] == 4:
            latents = image
        else:
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                        )
                init_latents = [retrieve_latents(self.vae.encode(image[i:i +
                    1]), generator=generator[i]) for i in range(batch_size)]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image),
                    generator=generator)
            if self.vae.config.force_upcast:
                self.vae.to(dtype)
            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents
            latents = randn_tensor(shape, generator=generator, device=
                device, dtype=dtype)
            latents = latents * self.scheduler.init_noise_sigma
            if latent_interpolation_method == 'lerp':

                def latent_cls(v0, v1, index):
                    return lerp(v0, v1, index / num_frames * (1 - strength))
            elif latent_interpolation_method == 'slerp':

                def latent_cls(v0, v1, index):
                    return slerp(v0, v1, index / num_frames * (1 - strength))
            else:
                latent_cls = latent_interpolation_method
            for i in range(num_frames):
                latents[:, :, i, :, :] = latent_cls(latents[:, :, i, :, :],
                    init_latents, i)
    else:
        if shape != latents.shape:
            raise ValueError(
                f'`latents` expected to have shape={shape!r}, but found latents.shape={latents.shape!r}'
                )
        latents = latents.to(device, dtype=dtype)
    return latents
