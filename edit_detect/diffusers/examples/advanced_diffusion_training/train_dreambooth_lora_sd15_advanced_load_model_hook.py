def load_model_hook(models, input_dir):
    unet_ = None
    text_encoder_one_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(accelerator.unwrap_model(unet))):
            unet_ = model
        elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))
            ):
            text_encoder_one_ = model
        else:
            raise ValueError(f'unexpected save model: {model.__class__}')
    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir
        )
    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=
        network_alphas, unet=unet_)
    text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if 
        'text_encoder.' in k}
    LoraLoaderMixin.load_lora_into_text_encoder(text_encoder_state_dict,
        network_alphas=network_alphas, text_encoder=text_encoder_one_)
