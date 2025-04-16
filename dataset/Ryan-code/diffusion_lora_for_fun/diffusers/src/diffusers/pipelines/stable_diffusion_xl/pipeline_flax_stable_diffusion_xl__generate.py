def _generate(self, prompt_ids: jnp.array, params: Union[Dict, FrozenDict],
    prng_seed: jax.Array, num_inference_steps: int, height: int, width: int,
    guidance_scale: float, latents: Optional[jnp.array]=None,
    neg_prompt_ids: Optional[jnp.array]=None, return_latents=False):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    prompt_embeds, pooled_embeds = self.get_embeddings(prompt_ids, params)
    batch_size = prompt_embeds.shape[0]
    if neg_prompt_ids is None:
        neg_prompt_embeds = jnp.zeros_like(prompt_embeds)
        negative_pooled_embeds = jnp.zeros_like(pooled_embeds)
    else:
        neg_prompt_embeds, negative_pooled_embeds = self.get_embeddings(
            neg_prompt_ids, params)
    add_time_ids = self._get_add_time_ids((height, width), (0, 0), (height,
        width), prompt_embeds.shape[0], dtype=prompt_embeds.dtype)
    prompt_embeds = jnp.concatenate([neg_prompt_embeds, prompt_embeds], axis=0)
    add_text_embeds = jnp.concatenate([negative_pooled_embeds,
        pooled_embeds], axis=0)
    add_time_ids = jnp.concatenate([add_time_ids, add_time_ids], axis=0)
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
    latents_shape = (batch_size, self.unet.config.in_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=
            jnp.float32)
    elif latents.shape != latents_shape:
        raise ValueError(
            f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
            )
    scheduler_state = self.scheduler.set_timesteps(params['scheduler'],
        num_inference_steps=num_inference_steps, shape=latents.shape)
    latents = latents * scheduler_state.init_noise_sigma
    added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
        add_time_ids}

    def loop_body(step, args):
        latents, scheduler_state = args
        latents_input = jnp.concatenate([latents] * 2)
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
        latents_input = self.scheduler.scale_model_input(scheduler_state,
            latents_input, t)
        noise_pred = self.unet.apply({'params': params['unet']}, jnp.array(
            latents_input), jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=prompt_embeds, added_cond_kwargs=
            added_cond_kwargs).sample
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2,
            axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        latents, scheduler_state = self.scheduler.step(scheduler_state,
            noise_pred, t, latents).to_tuple()
        return latents, scheduler_state
    if DEBUG:
        for i in range(num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (
            latents, scheduler_state))
    if return_latents:
        return latents
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.apply({'params': params['vae']}, latents, method=self.
        vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image
