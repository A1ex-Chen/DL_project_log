def eval_model(model, vae, stft, stage, eval_dataloader, accelerator,
    num_diffusion_steps):
    model.eval()
    model.uncondition = False
    num_data_to_eval = len(eval_dataloader.dataset) if stage == 1 else 100
    num_losses = 4 if stage >= 2 else 1
    total_val_losses = [[] for _ in range(num_losses)]
    eval_steps = [num_diffusion_steps - 1]
    eval_progress_bar = tqdm(range(num_data_to_eval * len(eval_steps)),
        disable=not accelerator.is_local_main_process)
    for validation_mode in eval_steps:
        total_val_loss = [(0) for _ in range(num_losses)]
        num_tested = 0
        for cntr, (captions, gt_waves, _) in enumerate(eval_dataloader):
            with (accelerator.accumulate(model) and torch.no_grad()):
                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _ = torch_tools.wav_to_fbank(gt_waves, TARGET_LENGTH, stft
                    )
                mel = mel.unsqueeze(1).to(model.device)
                true_latent = unwrapped_vae.get_first_stage_encoding(
                    unwrapped_vae.encode_first_stage(mel))
                val_loss = model(true_latent, gt_waves, captions,
                    validation_mode=validation_mode)
                eval_progress_bar.update(len(captions) * torch.cuda.
                    device_count())
                if not isinstance(val_loss, tuple) and not isinstance(val_loss,
                    list):
                    val_loss = [val_loss]
                for i in range(num_losses):
                    total_val_loss[i] += val_loss[i].detach().float().item()
                num_tested += len(captions) * torch.cuda.device_count()
                if num_tested >= num_data_to_eval:
                    break
        accelerator.print()
        for i in range(num_losses):
            total_val_losses[i] += [total_val_loss[i] / cntr]
            logger.info(
                f'{validation_mode} steps loss {i + 1}: {total_val_losses[i][-1]}'
                )
    return [np.array(tvl).mean() for tvl in total_val_losses]
