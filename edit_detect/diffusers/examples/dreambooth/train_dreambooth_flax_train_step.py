def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
    if args.train_text_encoder:
        params = {'text_encoder': text_encoder_state.params, 'unet':
            unet_state.params}
    else:
        params = {'unet': unet_state.params}

    def compute_loss(params):
        vae_outputs = vae.apply({'params': vae_params}, batch[
            'pixel_values'], deterministic=True, method=vae.encode)
        latents = vae_outputs.latent_dist.sample(sample_rng)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        latents = latents * vae.config.scaling_factor
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        bsz = latents.shape[0]
        timesteps = jax.random.randint(timestep_rng, (bsz,), 0,
            noise_scheduler.config.num_train_timesteps)
        noisy_latents = noise_scheduler.add_noise(noise_scheduler_state,
            latents, noise, timesteps)
        if args.train_text_encoder:
            encoder_hidden_states = text_encoder_state.apply_fn(batch[
                'input_ids'], params=params['text_encoder'], dropout_rng=
                dropout_rng, train=True)[0]
        else:
            encoder_hidden_states = text_encoder(batch['input_ids'], params
                =text_encoder_state.params, train=False)[0]
        model_pred = unet.apply({'params': params['unet']}, noisy_latents,
            timesteps, encoder_hidden_states, train=True).sample
        if noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif noise_scheduler.config.prediction_type == 'v_prediction':
            target = noise_scheduler.get_velocity(noise_scheduler_state,
                latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                )
        if args.with_prior_preservation:
            model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
            target, target_prior = jnp.split(target, 2, axis=0)
            loss = (target - model_pred) ** 2
            loss = loss.mean()
            prior_loss = (target_prior - model_pred_prior) ** 2
            prior_loss = prior_loss.mean()
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = (target - model_pred) ** 2
            loss = loss.mean()
        return loss
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(params)
    grad = jax.lax.pmean(grad, 'batch')
    new_unet_state = unet_state.apply_gradients(grads=grad['unet'])
    if args.train_text_encoder:
        new_text_encoder_state = text_encoder_state.apply_gradients(grads=
            grad['text_encoder'])
    else:
        new_text_encoder_state = text_encoder_state
    metrics = {'loss': loss}
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return new_unet_state, new_text_encoder_state, metrics, new_train_rng