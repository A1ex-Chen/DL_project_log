def load_model_hook(models, input_dir):
    load_model = UNet2DConditionModel.from_pretrained(os.path.join(
        input_dir, 'unet_target'))
    target_unet.load_state_dict(load_model.state_dict())
    target_unet.to(accelerator.device)
    del load_model
    for i in range(len(models)):
        model = models.pop()
        load_model = UNet2DConditionModel.from_pretrained(input_dir,
            subfolder='unet')
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model
