def load_model_hook(models, input_dir):
    transformer = None
    text_encoder_ = None
    while len(models) > 0:
        model_ = models.pop()
        if isinstance(model_, type(accelerator.unwrap_model(model))):
            if args.use_lora:
                transformer = model_
            else:
                load_model = UVit2DModel.from_pretrained(os.path.join(
                    input_dir, 'transformer'))
                model_.load_state_dict(load_model.state_dict())
                del load_model
        elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
            if args.text_encoder_use_lora:
                text_encoder_ = model_
            else:
                load_model = CLIPTextModelWithProjection.from_pretrained(os
                    .path.join(input_dir, 'text_encoder'))
                model_.load_state_dict(load_model.state_dict())
                del load_model
        else:
            raise ValueError(f'unexpected save model: {model.__class__}')
    if transformer is not None or text_encoder_ is not None:
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
            input_dir)
        LoraLoaderMixin.load_lora_into_text_encoder(lora_state_dict,
            network_alphas=network_alphas, text_encoder=text_encoder_)
        LoraLoaderMixin.load_lora_into_transformer(lora_state_dict,
            network_alphas=network_alphas, transformer=transformer)
    if args.use_ema:
        load_from = EMAModel.from_pretrained(os.path.join(input_dir,
            'ema_model'), model_cls=UVit2DModel)
        ema.load_state_dict(load_from.state_dict())
        del load_from
