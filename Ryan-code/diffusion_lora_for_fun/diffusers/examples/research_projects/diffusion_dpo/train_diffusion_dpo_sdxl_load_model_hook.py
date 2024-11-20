def load_model_hook(models, input_dir):
    unet_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(accelerator.unwrap_model(unet))):
            unet_ = model
        else:
            raise ValueError(f'unexpected save model: {model.__class__}')
    lora_state_dict, network_alphas = (StableDiffusionXLLoraLoaderMixin.
        lora_state_dict(input_dir))
    StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(lora_state_dict,
        network_alphas=network_alphas, unet=unet_)
