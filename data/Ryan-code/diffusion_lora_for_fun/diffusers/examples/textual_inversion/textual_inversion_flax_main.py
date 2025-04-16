def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR
        )
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer')
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f'The tokenizer already contains the token {args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.'
            )
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens
        =False)
    if len(token_ids) > 1:
        raise ValueError('The initializer token must be a single token.')
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.
        placeholder_token)
    text_encoder = FlaxCLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='vae', revision=args.revision)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    rng = jax.random.PRNGKey(args.seed)
    rng, _ = jax.random.split(rng)
    text_encoder = resize_token_embeddings(text_encoder, len(tokenizer),
        initializer_token_id, placeholder_token_id, rng)
    original_token_embeds = text_encoder.params['text_model']['embeddings'][
        'token_embedding']['embedding']
    train_dataset = TextualInversionDataset(data_root=args.train_data_dir,
        tokenizer=tokenizer, size=args.resolution, placeholder_token=args.
        placeholder_token, repeats=args.repeats, learnable_property=args.
        learnable_property, center_crop=args.center_crop, set='train')

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in
            examples])
        input_ids = torch.stack([example['input_ids'] for example in examples])
        batch = {'pixel_values': pixel_values, 'input_ids': input_ids}
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch
    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=total_train_batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn)
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    constant_scheduler = optax.constant_schedule(args.learning_rate)
    optimizer = optax.adamw(learning_rate=constant_scheduler, b1=args.
        adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay
        =args.adam_weight_decay)

    def create_mask(params, label_fn):

        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = 'token_embedding'
                elif isinstance(params[k], dict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'zero'
        mask = {}
        _map(params, mask, label_fn)
        return mask

    def zero_grads():

        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.tree_util.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)
    tx = optax.multi_transform({'token_embedding': optimizer, 'zero':
        zero_grads()}, create_mask(text_encoder.params, lambda s: s ==
        'token_embedding'))
    state = train_state.TrainState.create(apply_fn=text_encoder.__call__,
        params=text_encoder.params, tx=tx)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    train_rngs = jax.random.split(rng, jax.local_device_count())

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
            encoder_hidden_states = state.apply_fn(batch['input_ids'],
                params=params, dropout_rng=dropout_rng, train=True)[0]
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
        token_embeds = original_token_embeds.at[placeholder_token_id].set(
            new_state.params['text_model']['embeddings']['token_embedding']
            ['embedding'][placeholder_token_id])
        new_state.params['text_model']['embeddings']['token_embedding'][
            'embedding'] = token_embeds
        metrics = {'loss': loss}
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return new_state, metrics, new_train_rng
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,))
    state = jax_utils.replicate(state)
    vae_params = jax_utils.replicate(vae_params)
    unet_params = jax_utils.replicate(unet_params)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel & distributed) = {total_train_batch_size}'
        )
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0
    epochs = tqdm(range(args.num_train_epochs), desc=
        f'Epoch ... (1/{args.num_train_epochs})', position=0)
    for epoch in epochs:
        train_metrics = []
        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc=
            'Training...', position=1, leave=False)
        for batch in train_dataloader:
            batch = shard(batch)
            state, train_metric, train_rngs = p_train_step(state,
                vae_params, unet_params, batch, train_rngs)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)
            global_step += 1
            if global_step >= args.max_train_steps:
                break
            if global_step % args.save_steps == 0:
                learned_embeds = get_params_to_save(state.params)['text_model'
                    ]['embeddings']['token_embedding']['embedding'][
                    placeholder_token_id]
                learned_embeds_dict = {args.placeholder_token: learned_embeds}
                jnp.save(os.path.join(args.output_dir, 'learned_embeds-' +
                    str(global_step) + '.npy'), learned_embeds_dict)
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})"
            )
    if jax.process_index() == 0:
        scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', skip_prk_steps=True)
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker', from_pt=True)
        pipeline = FlaxStableDiffusionPipeline(text_encoder=text_encoder,
            vae=vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=
            CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32'))
        pipeline.save_pretrained(args.output_dir, params={'text_encoder':
            get_params_to_save(state.params), 'vae': get_params_to_save(
            vae_params), 'unet': get_params_to_save(unet_params),
            'safety_checker': safety_checker.params})
        learned_embeds = get_params_to_save(state.params)['text_model'][
            'embeddings']['token_embedding']['embedding'][placeholder_token_id]
        learned_embeds_dict = {args.placeholder_token: learned_embeds}
        jnp.save(os.path.join(args.output_dir, 'learned_embeds.npy'),
            learned_embeds_dict)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
