def load_model_hook(models, input_dir):
    load_model = UNet2DModel.from_pretrained(os.path.join(input_dir,
        'unet_teacher'))
    teacher_unet.load_state_dict(load_model.state_dict())
    teacher_unet.to(accelerator.device)
    del load_model
    if args.use_ema:
        load_model = EMAModel.from_pretrained(os.path.join(input_dir,
            'unet_ema'), UNet2DModel)
        ema_unet.load_state_dict(load_model.state_dict())
        ema_unet.to(accelerator.device)
        del load_model
    for i in range(len(models)):
        model = models.pop()
        load_model = UNet2DModel.from_pretrained(input_dir, subfolder='unet')
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model
