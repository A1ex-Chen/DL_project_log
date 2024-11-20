def _generate(self, prompt_ids: jnp.array, mask: jnp.array, masked_image:
    jnp.array, params: Union[Dict, FrozenDict], prng_seed: jax.random.
    KeyArray, num_inference_steps: int, height: int, width: int,
    guidance_scale: float, latents: Optional[jnp.array]=None,
    neg_prompt_ids: Optional[jnp.array]=None):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    prompt_embeds = self.text_encoder(prompt_ids, params=params['text_encoder']
        )[0]
    batch_size = prompt_ids.shape[0]
    max_length = prompt_ids.shape[-1]
    if neg_prompt_ids is None:
        uncond_input = self.tokenizer([''] * batch_size, padding=
            'max_length', max_length=max_length, return_tensors='np').input_ids
    else:
        uncond_input = neg_prompt_ids
    negative_prompt_embeds = self.text_encoder(uncond_input, params=params[
        'text_encoder'])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
    latents_shape = (batch_size, self.vae.config.latent_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=
            self.dtype)
    elif latents.shape != latents_shape:
        raise ValueError(
            f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
            )
    prng_seed, mask_prng_seed = jax.random.split(prng_seed)
    masked_image_latent_dist = self.vae.apply({'params': params['vae']},
        masked_image, method=self.vae.encode).latent_dist
    masked_image_latents = masked_image_latent_dist.sample(key=mask_prng_seed
        ).transpose((0, 3, 1, 2))
    masked_image_latents = (self.vae.config.scaling_factor *
        masked_image_latents)
    del mask_prng_seed
    mask = jax.image.resize(mask, (*mask.shape[:-2], *masked_image_latents.
        shape[-2:]), method='nearest')
    num_channels_latents = self.vae.config.latent_channels
    num_channels_mask = mask.shape[1]
    num_channels_masked_image = masked_image_latents.shape[1]
    if (num_channels_latents + num_channels_mask +
        num_channels_masked_image != self.unet.config.in_channels):
        raise ValueError(
            f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.'
            )

    def loop_body(step, args):
        latents, mask, masked_image_latents, scheduler_state = args
        latents_input = jnp.concatenate([latents] * 2)
        mask_input = jnp.concatenate([mask] * 2)
        masked_image_latents_input = jnp.concatenate([masked_image_latents] * 2
            )
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
        latents_input = self.scheduler.scale_model_input(scheduler_state,
            latents_input, t)
        latents_input = jnp.concatenate([latents_input, mask_input,
            masked_image_latents_input], axis=1)
        noise_pred = self.unet.apply({'params': params['unet']}, jnp.array(
            latents_input), jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context).sample
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2,
            axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        latents, scheduler_state = self.scheduler.step(scheduler_state,
            noise_pred, t, latents).to_tuple()
        return latents, mask, masked_image_latents, scheduler_state
    scheduler_state = self.scheduler.set_timesteps(params['scheduler'],
        num_inference_steps=num_inference_steps, shape=latents.shape)
    latents = latents * params['scheduler'].init_noise_sigma
    if DEBUG:
        for i in range(num_inference_steps):
            latents, mask, masked_image_latents, scheduler_state = loop_body(i,
                (latents, mask, masked_image_latents, scheduler_state))
    else:
        latents, _, _, _ = jax.lax.fori_loop(0, num_inference_steps,
            loop_body, (latents, mask, masked_image_latents, scheduler_state))
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.apply({'params': params['vae']}, latents, method=self.
        vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image
