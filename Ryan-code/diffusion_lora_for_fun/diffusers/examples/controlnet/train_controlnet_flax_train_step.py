def train_step(state, unet_params, text_encoder_params, vae_params, batch,
    train_rng):
    if args.gradient_accumulation_steps > 1:
        grad_steps = args.gradient_accumulation_steps
        batch = jax.tree_map(lambda x: x.reshape((grad_steps, x.shape[0] //
            grad_steps) + x.shape[1:]), batch)

    def compute_loss(params, minibatch, sample_rng):
        vae_outputs = vae.apply({'params': vae_params}, minibatch[
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
        encoder_hidden_states = text_encoder(minibatch['input_ids'], params
            =text_encoder_params, train=False)[0]
        controlnet_cond = minibatch['conditioning_pixel_values']
        down_block_res_samples, mid_block_res_sample = controlnet.apply({
            'params': params}, noisy_latents, timesteps,
            encoder_hidden_states, controlnet_cond, train=True, return_dict
            =False)
        model_pred = unet.apply({'params': unet_params}, noisy_latents,
            timesteps, encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample).sample
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
    grad_fn = jax.value_and_grad(compute_loss)

    def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(lambda x: jax.lax.
            dynamic_index_in_dim(x, grad_idx, keepdims=False), batch)

    def loss_and_grad(grad_idx, train_rng):
        minibatch = get_minibatch(batch, grad_idx
            ) if grad_idx is not None else batch
        sample_rng, train_rng = jax.random.split(train_rng, 2)
        loss, grad = grad_fn(state.params, minibatch, sample_rng)
        return loss, grad, train_rng
    if args.gradient_accumulation_steps == 1:
        loss, grad, new_train_rng = loss_and_grad(None, train_rng)
    else:
        init_loss_grad_rng = 0.0, jax.tree_map(jnp.zeros_like, state.params
            ), train_rng

        def cumul_grad_step(grad_idx, loss_grad_rng):
            cumul_loss, cumul_grad, train_rng = loss_grad_rng
            loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
            cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss,
                cumul_grad), (loss, grad))
            return cumul_loss, cumul_grad, new_train_rng
        loss, grad, new_train_rng = jax.lax.fori_loop(0, args.
            gradient_accumulation_steps, cumul_grad_step, init_loss_grad_rng)
        loss, grad = jax.tree_map(lambda x: x / args.
            gradient_accumulation_steps, (loss, grad))
    grad = jax.lax.pmean(grad, 'batch')
    new_state = state.apply_gradients(grads=grad)
    metrics = {'loss': loss}
    metrics = jax.lax.pmean(metrics, axis_name='batch')

    def l2(xs):
        return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.
            tree_leaves(xs)]))
    metrics['l2_grads'] = l2(jax.tree_util.tree_leaves(grad))
    return new_state, metrics, new_train_rng
