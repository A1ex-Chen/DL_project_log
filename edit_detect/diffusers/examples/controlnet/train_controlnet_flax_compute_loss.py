def compute_loss(params, minibatch, sample_rng):
    vae_outputs = vae.apply({'params': vae_params}, minibatch[
        'pixel_values'], deterministic=True, method=vae.encode)
    latents = vae_outputs.latent_dist.sample(sample_rng)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * vae.config.scaling_factor
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)
    bsz = latents.shape[0]
    timesteps = jax.random.randint(timestep_rng, (bsz,), 0, noise_scheduler
        .config.num_train_timesteps)
    noisy_latents = noise_scheduler.add_noise(noise_scheduler_state,
        latents, noise, timesteps)
    encoder_hidden_states = text_encoder(minibatch['input_ids'], params=
        text_encoder_params, train=False)[0]
    controlnet_cond = minibatch['conditioning_pixel_values']
    down_block_res_samples, mid_block_res_sample = controlnet.apply({
        'params': params}, noisy_latents, timesteps, encoder_hidden_states,
        controlnet_cond, train=True, return_dict=False)
    model_pred = unet.apply({'params': unet_params}, noisy_latents,
        timesteps, encoder_hidden_states, down_block_additional_residuals=
        down_block_res_samples, mid_block_additional_residual=
        mid_block_res_sample).sample
    if noise_scheduler.config.prediction_type == 'epsilon':
        target = noise
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        target = noise_scheduler.get_velocity(noise_scheduler_state,
            latents, noise, timesteps)
    else:
        raise ValueError(
            f'Unknown prediction type {noise_scheduler.config.prediction_type}'
            )
    loss = (target - model_pred) ** 2
    if args.snr_gamma is not None:
        snr = jnp.array(compute_snr(timesteps))
        snr_loss_weights = jnp.where(snr < args.snr_gamma, snr, jnp.
            ones_like(snr) * args.snr_gamma)
        if noise_scheduler.config.prediction_type == 'epsilon':
            snr_loss_weights = snr_loss_weights / snr
        elif noise_scheduler.config.prediction_type == 'v_prediction':
            snr_loss_weights = snr_loss_weights / (snr + 1)
        loss = loss * snr_loss_weights
    loss = loss.mean()
    return loss
