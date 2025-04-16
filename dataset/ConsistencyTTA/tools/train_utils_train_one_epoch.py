def train_one_epoch(model, vae, stft, train_dataloader, accelerator, args,
    optimizer, lr_scheduler, checkpointing_steps, completed_steps, progress_bar
    ):
    model.train()
    model.uncondition = args.uncondition
    total_loss = 0
    for captions, gt_waves, _ in train_dataloader:
        with accelerator.accumulate(model):
            with torch.no_grad():
                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _ = torch_tools.wav_to_fbank(gt_waves, TARGET_LENGTH, stft
                    )
                mel = mel.unsqueeze(1).to(model.device)
                true_x0 = unwrapped_vae.get_first_stage_encoding(unwrapped_vae
                    .encode_first_stage(mel))
            loss = model(true_x0, gt_waves, captions, validation_mode=False)
            accelerator.backward(loss)
            if not torch.isnan(loss):
                total_loss += loss.detach().float().item()
                optimizer.step()
                lr_scheduler.step()
            else:
                logger.info('NaN loss encountered.')
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            with torch.no_grad():
                if torch.cuda.device_count() == 1:
                    model.update_ema()
                else:
                    model.module.update_ema()
            progress_bar.update(1)
            progress_bar.set_postfix({'lr': optimizer.param_groups[0]['lr'],
                'train_loss': loss.item()})
            completed_steps += 1
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f'step_{completed_steps}'
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
        if completed_steps >= args.max_train_steps:
            break
    return total_loss, completed_steps
