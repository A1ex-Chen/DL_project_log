def _generate(self, prompt_ids: jnp.ndarray, image: jnp.ndarray, params:
    Union[Dict, FrozenDict], prng_seed: jax.Array, num_inference_steps: int,
    guidance_scale: float, latents: Optional[jnp.ndarray]=None,
    neg_prompt_ids: Optional[jnp.ndarray]=None,
    controlnet_conditioning_scale: float=1.0):
    height, width = image.shape[-2:]
    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 64 but are {height} and {width}.'
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
    image = jnp.concatenate([image] * 2)
    latents_shape = (batch_size, self.unet.config.in_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=
            jnp.float32)
    elif latents.shape != latents_shape:
        raise ValueError(
            f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
            )

    def loop_body(step, args):
        latents, scheduler_state = args
        latents_input = jnp.concatenate([latents] * 2)
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
        latents_input = self.scheduler.scale_model_input(scheduler_state,
            latents_input, t)
        down_block_res_samples, mid_block_res_sample = self.controlnet.apply({
            'params': params['controlnet']}, jnp.array(latents_input), jnp.
            array(timestep, dtype=jnp.int32), encoder_hidden_states=context,
            controlnet_cond=image, conditioning_scale=
            controlnet_conditioning_scale, return_dict=False)
        noise_pred = self.unet.apply({'params': params['unet']}, jnp.array(
            latents_input), jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context, down_block_additional_residuals=
            down_block_res_samples, mid_block_additional_residual=
            mid_block_res_sample).sample
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2,
            axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        latents, scheduler_state = self.scheduler.step(scheduler_state,
            noise_pred, t, latents).to_tuple()
        return latents, scheduler_state
    scheduler_state = self.scheduler.set_timesteps(params['scheduler'],
        num_inference_steps=num_inference_steps, shape=latents_shape)
    latents = latents * params['scheduler'].init_noise_sigma
    if DEBUG:
        for i in range(num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (
            latents, scheduler_state))
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.apply({'params': params['vae']}, latents, method=self.
        vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image
