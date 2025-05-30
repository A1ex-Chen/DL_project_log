def train_step(state, vae_params, unet_params, batch, train_rng):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

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
        encoder_hidden_states = state.apply_fn(batch['input_ids'], params=
            params, dropout_rng=dropout_rng, train=True)[0]
        model_pred = unet.apply({'params': unet_params}, noisy_latents,
            timesteps, encoder_hidden_states, train=False).sample
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
        loss = loss.mean()
        return loss
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, 'batch')
    new_state = state.apply_gradients(grads=grad)
    token_embeds = original_token_embeds.at[placeholder_token_id].set(new_state
        .params['text_model']['embeddings']['token_embedding']['embedding']
        [placeholder_token_id])
    new_state.params['text_model']['embeddings']['token_embedding']['embedding'
        ] = token_embeds
    metrics = {'loss': loss}
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return new_state, metrics, new_train_rng
