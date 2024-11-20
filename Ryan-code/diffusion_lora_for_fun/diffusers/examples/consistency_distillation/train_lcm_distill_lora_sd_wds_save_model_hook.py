def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unet_, adapter_name=
            'default')
        StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir,
            'unet_lora'), lora_state_dict)
        unet_.save_pretrained(output_dir)
        for _, model in enumerate(models):
            weights.pop()
