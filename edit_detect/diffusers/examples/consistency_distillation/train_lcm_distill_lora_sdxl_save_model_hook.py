def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict
            (unet_))
        StableDiffusionXLPipeline.save_lora_weights(output_dir,
            unet_lora_layers=state_dict)
        for _, model in enumerate(models):
            weights.pop()
