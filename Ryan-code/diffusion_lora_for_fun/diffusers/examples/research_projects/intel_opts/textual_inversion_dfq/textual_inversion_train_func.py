def train_func(model):
    if train_unet:
        unet_ = model
        text_encoder_ = text_encoder
    else:
        unet_ = unet
        text_encoder_ = model
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('textual_inversion', config=vars(args))
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
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
                latents = vae.encode(batch['pixel_values']).latent_dist.sample(
                    ).detach()
                latents = latents * 0.18215
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states = text_encoder_(batch['input_ids'])[0]
                model_pred = unet_(noisy_latents, timesteps,
                    encoder_hidden_states).sample
                loss = F.mse_loss(model_pred, noise, reduction='none').mean([
                    1, 2, 3]).mean()
                if train_unet and compression_manager:
                    unet_inputs = {'sample': noisy_latents, 'timestep':
                        timesteps, 'encoder_hidden_states':
                        encoder_hidden_states}
                    loss = compression_manager.callbacks.on_after_compute_loss(
                        unet_inputs, model_pred, loss)
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
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0
                if not train_unet and global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir,
                        f'learned_embeds-steps-{global_step}.bin')
                    save_progress(text_encoder_, placeholder_token_id,
                        accelerator, args, save_path)
            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    if train_unet and args.use_ema:
        ema_unet.copy_to(unet_.parameters())
    if not train_unet:
        return text_encoder_
