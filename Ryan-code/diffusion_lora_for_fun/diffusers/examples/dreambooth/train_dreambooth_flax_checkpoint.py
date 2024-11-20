def checkpoint(step=None):
    scheduler, _ = FlaxPNDMScheduler.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='scheduler')
    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker', from_pt=True)
    pipeline = FlaxStableDiffusionPipeline(text_encoder=text_encoder, vae=
        vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler,
        safety_checker=safety_checker, feature_extractor=CLIPImageProcessor
        .from_pretrained('openai/clip-vit-base-patch32'))
    outdir = os.path.join(args.output_dir, str(step)
        ) if step else args.output_dir
    pipeline.save_pretrained(outdir, params={'text_encoder':
        get_params_to_save(text_encoder_state.params), 'vae':
        get_params_to_save(vae_params), 'unet': get_params_to_save(
        unet_state.params), 'safety_checker': safety_checker.params})
    if args.push_to_hub:
        message = (f'checkpoint-{step}' if step is not None else
            'End of training')
        upload_folder(repo_id=repo_id, folder_path=args.output_dir,
            commit_message=message, ignore_patterns=['step_*', 'epoch_*'])
