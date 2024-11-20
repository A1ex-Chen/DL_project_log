def _generate(self, prompt_ids: jnp.ndarray, image: jnp.ndarray, params:
    Union[Dict, FrozenDict], prng_seed: jax.Array, start_timestep: int,
    num_inference_steps: int, height: int, width: int, guidance_scale:
    float, noise: Optional[jnp.ndarray]=None, neg_prompt_ids: Optional[jnp.
    ndarray]=None):
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
    latents_shape = (batch_size, self.unet.config.in_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor)
    if noise is None:
        noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp
            .float32)
    elif noise.shape != latents_shape:
        raise ValueError(
            f'Unexpected latents shape, got {noise.shape}, expected {latents_shape}'
            )
    init_latent_dist = self.vae.apply({'params': params['vae']}, image,
        method=self.vae.encode).latent_dist
    init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 
        1, 2))
    init_latents = self.vae.config.scaling_factor * init_latents

    def loop_body(step, args):
        latents, scheduler_state = args
        latents_input = jnp.concatenate([latents] * 2)
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
        latents_input = self.scheduler.scale_model_input(scheduler_state,
            latents_input, t)
        noise_pred = self.unet.apply({'params': params['unet']}, jnp.array(
            latents_input), jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context).sample
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2,
            axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        latents, scheduler_state = self.scheduler.step(scheduler_state,
            noise_pred, t, latents).to_tuple()
        return latents, scheduler_state
    scheduler_state = self.scheduler.set_timesteps(params['scheduler'],
        num_inference_steps=num_inference_steps, shape=latents_shape)
    latent_timestep = scheduler_state.timesteps[start_timestep:
        start_timestep + 1].repeat(batch_size)
    latents = self.scheduler.add_noise(params['scheduler'], init_latents,
        noise, latent_timestep)
    latents = latents * params['scheduler'].init_noise_sigma
    if DEBUG:
        for i in range(start_timestep, num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps,
            loop_body, (latents, scheduler_state))
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.apply({'params': params['vae']}, latents, method=self.
        vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image
