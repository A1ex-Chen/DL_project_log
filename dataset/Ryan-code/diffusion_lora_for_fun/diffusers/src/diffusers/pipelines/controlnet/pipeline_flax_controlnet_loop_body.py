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
    noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text -
        noise_pred_uncond)
    latents, scheduler_state = self.scheduler.step(scheduler_state,
        noise_pred, t, latents).to_tuple()
    return latents, scheduler_state
