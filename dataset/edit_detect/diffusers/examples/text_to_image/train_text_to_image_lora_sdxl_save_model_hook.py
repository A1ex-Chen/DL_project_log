def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model))
            elif isinstance(unwrap_model(model), type(unwrap_model(
                text_encoder_one))):
                text_encoder_one_lora_layers_to_save = (
                    convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)))
            elif isinstance(unwrap_model(model), type(unwrap_model(
                text_encoder_two))):
                text_encoder_two_lora_layers_to_save = (
                    convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)))
            else:
                raise ValueError(f'unexpected save model: {model.__class__}')
            if weights:
                weights.pop()
        StableDiffusionXLPipeline.save_lora_weights(output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save)
