def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.
        output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with='tensorboard', project_config=accelerator_project_config)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_config(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    train_unet = False
    freeze_params(vae.parameters())
    if not args.do_quantization and not args.do_distillation:
        num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f'The tokenizer already contains the token {args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.'
                )
        token_ids = tokenizer.encode(args.initializer_token,
            add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError('The initializer token must be a single token.')
        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(args.
            placeholder_token)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
        freeze_params(unet.parameters())
        params_to_freeze = itertools.chain(text_encoder.text_model.encoder.
            parameters(), text_encoder.text_model.final_layer_norm.
            parameters(), text_encoder.text_model.embeddings.
            position_embedding.parameters())
        freeze_params(params_to_freeze)
    else:
        train_unet = True
        freeze_params(text_encoder.parameters())
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    optimizer = torch.optim.AdamW(unet.parameters() if train_unet else
        text_encoder.get_input_embeddings().parameters(), lr=args.
        learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    train_dataset = TextualInversionDataset(data_root=args.train_data_dir,
        tokenizer=tokenizer, size=args.resolution, placeholder_token=args.
        placeholder_token, repeats=args.repeats, learnable_property=args.
        learnable_property, center_crop=args.center_crop, set='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)
    if not train_unet:
        text_encoder = accelerator.prepare(text_encoder)
        unet.to(accelerator.device)
        unet.eval()
    else:
        unet = accelerator.prepare(unet)
        text_encoder.to(accelerator.device)
        text_encoder.eval()
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer,
        train_dataloader, lr_scheduler)
    vae.to(accelerator.device)
    vae.eval()
    compression_manager = None

    def train_func(model):
        if train_unet:
            unet_ = model
            text_encoder_ = text_encoder
        else:
            unet_ = unet
            text_encoder_ = model
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args
            .gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = (args.num_train_epochs *
                num_update_steps_per_epoch)
        args.num_train_epochs = math.ceil(args.max_train_steps /
            num_update_steps_per_epoch)
        if accelerator.is_main_process:
            accelerator.init_trackers('textual_inversion', config=vars(args))
        total_batch_size = (args.train_batch_size * accelerator.
            num_processes * args.gradient_accumulation_steps)
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {len(train_dataset)}')
        logger.info(f'  Num Epochs = {args.num_train_epochs}')
        logger.info(
            f'  Instantaneous batch size per device = {args.train_batch_size}')
        logger.info(
            f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
            )
        logger.info(
            f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}'
            )
        logger.info(f'  Total optimization steps = {args.max_train_steps}')
        progress_bar = tqdm(range(args.max_train_steps), disable=not
            accelerator.is_local_main_process)
        progress_bar.set_description('Steps')
        global_step = 0
        if train_unet and args.use_ema:
            ema_unet = EMAModel(unet_.parameters())
        for epoch in range(args.num_train_epochs):
            model.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    latents = vae.encode(batch['pixel_values']
                        ).latent_dist.sample().detach()
                    latents = latents * 0.18215
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.
                        num_train_timesteps, (bsz,), device=latents.device
                        ).long()
                    noisy_latents = noise_scheduler.add_noise(latents,
                        noise, timesteps)
                    encoder_hidden_states = text_encoder_(batch['input_ids'])[0
                        ]
                    model_pred = unet_(noisy_latents, timesteps,
                        encoder_hidden_states).sample
                    loss = F.mse_loss(model_pred, noise, reduction='none'
                        ).mean([1, 2, 3]).mean()
                    if train_unet and compression_manager:
                        unet_inputs = {'sample': noisy_latents, 'timestep':
                            timesteps, 'encoder_hidden_states':
                            encoder_hidden_states}
                        loss = (compression_manager.callbacks.
                            on_after_compute_loss(unet_inputs, model_pred,
                            loss))
                    avg_loss = accelerator.gather(loss.repeat(args.
                        train_batch_size)).mean()
                    train_loss += avg_loss.item(
                        ) / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if train_unet:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet_.parameters(),
                                args.max_grad_norm)
                    else:
                        if accelerator.num_processes > 1:
                            grads = text_encoder_.module.get_input_embeddings(
                                ).weight.grad
                        else:
                            grads = text_encoder_.get_input_embeddings(
                                ).weight.grad
                        index_grads_to_zero = torch.arange(len(tokenizer)
                            ) != placeholder_token_id
                        grads.data[index_grads_to_zero, :] = grads.data[
                            index_grads_to_zero, :].fill_(0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if train_unet and args.use_ema:
                        ema_unet.step(unet_.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({'train_loss': train_loss}, step=
                        global_step)
                    train_loss = 0.0
                    if not train_unet and global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir,
                            f'learned_embeds-steps-{global_step}.bin')
                        save_progress(text_encoder_, placeholder_token_id,
                            accelerator, args, save_path)
                logs = {'step_loss': loss.detach().item(), 'lr':
                    lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if global_step >= args.max_train_steps:
                    break
            accelerator.wait_for_everyone()
        if train_unet and args.use_ema:
            ema_unet.copy_to(unet_.parameters())
        if not train_unet:
            return text_encoder_
    if not train_unet:
        text_encoder = train_func(text_encoder)
    else:
        import copy
        model = copy.deepcopy(unet)
        confs = []
        if args.do_quantization:
            from neural_compressor import QuantizationAwareTrainingConfig
            q_conf = QuantizationAwareTrainingConfig()
            confs.append(q_conf)
        if args.do_distillation:
            teacher_model = copy.deepcopy(model)

            def attention_fetcher(x):
                return x.sample
            layer_mappings = [[['conv_in']], [['time_embedding']], [[
                'down_blocks.0.attentions.0', attention_fetcher]], [[
                'down_blocks.0.attentions.1', attention_fetcher]], [[
                'down_blocks.0.resnets.0']], [['down_blocks.0.resnets.1']],
                [['down_blocks.0.downsamplers.0']], [[
                'down_blocks.1.attentions.0', attention_fetcher]], [[
                'down_blocks.1.attentions.1', attention_fetcher]], [[
                'down_blocks.1.resnets.0']], [['down_blocks.1.resnets.1']],
                [['down_blocks.1.downsamplers.0']], [[
                'down_blocks.2.attentions.0', attention_fetcher]], [[
                'down_blocks.2.attentions.1', attention_fetcher]], [[
                'down_blocks.2.resnets.0']], [['down_blocks.2.resnets.1']],
                [['down_blocks.2.downsamplers.0']], [[
                'down_blocks.3.resnets.0']], [['down_blocks.3.resnets.1']],
                [['up_blocks.0.resnets.0']], [['up_blocks.0.resnets.1']], [
                ['up_blocks.0.resnets.2']], [['up_blocks.0.upsamplers.0']],
                [['up_blocks.1.attentions.0', attention_fetcher]], [[
                'up_blocks.1.attentions.1', attention_fetcher]], [[
                'up_blocks.1.attentions.2', attention_fetcher]], [[
                'up_blocks.1.resnets.0']], [['up_blocks.1.resnets.1']], [[
                'up_blocks.1.resnets.2']], [['up_blocks.1.upsamplers.0']],
                [['up_blocks.2.attentions.0', attention_fetcher]], [[
                'up_blocks.2.attentions.1', attention_fetcher]], [[
                'up_blocks.2.attentions.2', attention_fetcher]], [[
                'up_blocks.2.resnets.0']], [['up_blocks.2.resnets.1']], [[
                'up_blocks.2.resnets.2']], [['up_blocks.2.upsamplers.0']],
                [['up_blocks.3.attentions.0', attention_fetcher]], [[
                'up_blocks.3.attentions.1', attention_fetcher]], [[
                'up_blocks.3.attentions.2', attention_fetcher]], [[
                'up_blocks.3.resnets.0']], [['up_blocks.3.resnets.1']], [[
                'up_blocks.3.resnets.2']], [['mid_block.attentions.0',
                attention_fetcher]], [['mid_block.resnets.0']], [[
                'mid_block.resnets.1']], [['conv_out']]]
            layer_names = [layer_mapping[0][0] for layer_mapping in
                layer_mappings]
            if not set(layer_names).issubset([n[0] for n in model.
                named_modules()]):
                raise ValueError(
                    f"""Provided model is not compatible with the default layer_mappings, please use the model fine-tuned from "CompVis/stable-diffusion-v1-4", or modify the layer_mappings variable to fit your model.
Default layer_mappings are as such:
{layer_mappings}"""
                    )
            from neural_compressor.config import DistillationConfig, IntermediateLayersKnowledgeDistillationLossConfig
            distillation_criterion = (
                IntermediateLayersKnowledgeDistillationLossConfig(
                layer_mappings=layer_mappings, loss_types=['MSE'] * len(
                layer_mappings), loss_weights=[1.0 / len(layer_mappings)] *
                len(layer_mappings), add_origin_loss=True))
            d_conf = DistillationConfig(teacher_model=teacher_model,
                criterion=distillation_criterion)
            confs.append(d_conf)
        from neural_compressor.training import prepare_compression
        compression_manager = prepare_compression(model, confs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        train_func(model)
        compression_manager.callbacks.on_train_end()
        model.save(args.output_dir)
        logger.info(f'Optimized model saved to: {args.output_dir}.')
        model = model.model
    templates = (imagenet_style_templates_small if args.learnable_property ==
        'style' else imagenet_templates_small)
    prompt = templates[0].format(args.placeholder_token)
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, text_encoder=accelerator.
            unwrap_model(text_encoder), vae=vae, unet=accelerator.
            unwrap_model(unet), tokenizer=tokenizer)
        pipeline.save_pretrained(args.output_dir)
        pipeline = pipeline.to(unet.device)
        baseline_model_images = generate_images(pipeline, prompt=prompt,
            seed=args.seed)
        baseline_model_images.save(os.path.join(args.output_dir,
            '{}_baseline_model.png'.format('_'.join(prompt.split()))))
        if not train_unet:
            save_path = os.path.join(args.output_dir, 'learned_embeds.bin')
            save_progress(text_encoder, placeholder_token_id, accelerator,
                args, save_path)
        else:
            setattr(pipeline, 'unet', accelerator.unwrap_model(model))
            if args.do_quantization:
                pipeline = pipeline.to(torch.device('cpu'))
            optimized_model_images = generate_images(pipeline, prompt=
                prompt, seed=args.seed)
            optimized_model_images.save(os.path.join(args.output_dir,
                '{}_optimized_model.png'.format('_'.join(prompt.split()))))
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
    if args.do_quantization and args.verify_loading:
        from neural_compressor.utils.pytorch import load
        loaded_model = load(args.output_dir, model=unet)
        loaded_model.eval()
        setattr(pipeline, 'unet', loaded_model)
        if args.do_quantization:
            pipeline = pipeline.to(torch.device('cpu'))
        loaded_model_images = generate_images(pipeline, prompt=prompt, seed
            =args.seed)
        if loaded_model_images != optimized_model_images:
            logger.info('The quantized model was not successfully loaded.')
        else:
            logger.info('The quantized model was successfully loaded.')
