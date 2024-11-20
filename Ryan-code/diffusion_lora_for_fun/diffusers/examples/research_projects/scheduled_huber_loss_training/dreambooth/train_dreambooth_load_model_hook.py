def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(unwrap_model(text_encoder))):
            load_model = text_encoder_cls.from_pretrained(input_dir,
                subfolder='text_encoder')
            model.config = load_model.config
        else:
            load_model = UNet2DConditionModel.from_pretrained(input_dir,
                subfolder='unet')
            model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        del load_model
