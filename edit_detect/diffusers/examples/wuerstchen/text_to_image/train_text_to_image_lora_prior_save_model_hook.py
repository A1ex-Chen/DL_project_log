def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        prior_lora_layers_to_save = None
        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(prior))):
                prior_lora_layers_to_save = get_peft_model_state_dict(model)
            else:
                raise ValueError(f'unexpected save model: {model.__class__}')
            weights.pop()
        WuerstchenPriorPipeline.save_lora_weights(output_dir,
            unet_lora_layers=prior_lora_layers_to_save)
