def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        sub_dir = 'unet' if isinstance(model.base_model.model, type(
            accelerator.unwrap_model(unet).base_model.model)
            ) else 'text_encoder'
        model_cls = UNet2DConditionModel if isinstance(model.base_model.
            model, type(accelerator.unwrap_model(unet).base_model.model)
            ) else CLIPTextModel
        load_model = model_cls.from_pretrained(args.
            pretrained_model_name_or_path, subfolder=sub_dir)
        load_model = PeftModel.from_pretrained(load_model, input_dir,
            subfolder=sub_dir)
        model.load_state_dict(load_model.state_dict())
        del load_model
