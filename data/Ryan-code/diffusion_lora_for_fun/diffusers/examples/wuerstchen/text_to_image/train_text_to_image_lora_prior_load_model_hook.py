def load_model_hook(models, input_dir):
    prior_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(accelerator.unwrap_model(prior))):
            prior_ = model
        else:
            raise ValueError(f'unexpected save model: {model.__class__}')
    lora_state_dict, network_alphas = WuerstchenPriorPipeline.lora_state_dict(
        input_dir)
    WuerstchenPriorPipeline.load_lora_into_unet(lora_state_dict,
        network_alphas=network_alphas, unet=prior_)
    WuerstchenPriorPipeline.load_lora_into_text_encoder(lora_state_dict,
        network_alphas=network_alphas)
