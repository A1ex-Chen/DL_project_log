def prepare_latents(self, image, mask, width, height, num_channels_latents,
    timestep, batch_size, num_images_per_prompt, dtype, device, generator=
    None, add_noise=True, latents=None, is_strength_max=True, return_noise=
    False, return_image_latents=False):
    batch_size *= num_images_per_prompt
    if image is None:
        shape = batch_size, num_channels_latents, int(height
            ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=
                device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    elif mask is None:
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
                )
        if hasattr(self, 'final_offload_hook'
            ) and self.final_offload_hook is not None:
            self.text_encoder_2.to('cpu')
            torch.cuda.empty_cache()
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] == 4:
            init_latents = image
        else:
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                    )
            elif isinstance(generator, list):
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
        if batch_size > init_latents.shape[0
            ] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] *
                additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0
            ] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f'Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.'
                )
        else:
            init_latents = torch.cat([init_latents], dim=0)
        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device,
                dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise,
                timestep)
        latents = init_latents
        return latents
    else:
        shape = batch_size, num_channels_latents, int(height
            ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                'Since strength < 1. initial latents are to be initialised as a combination of Image + Noise.However, either the image or the noise timestep has not been provided.'
                )
        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size //
                image_latents.shape[0], 1, 1, 1)
        elif return_image_latents or latents is None and not is_strength_max:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=
                generator)
            image_latents = image_latents.repeat(batch_size //
                image_latents.shape[0], 1, 1, 1)
        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=device,
                dtype=dtype)
            latents = noise if is_strength_max else self.scheduler.add_noise(
                image_latents, noise, timestep)
            latents = (latents * self.scheduler.init_noise_sigma if
                is_strength_max else latents)
        elif add_noise:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device,
                dtype=dtype)
            latents = image_latents.to(device)
        outputs = latents,
        if return_noise:
            outputs += noise,
        if return_image_latents:
            outputs += image_latents,
        return outputs
