def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        teacher_unet.save_pretrained(os.path.join(output_dir, 'unet_teacher'))
        if args.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, 'unet_ema'))
        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, 'unet'))
            weights.pop()
