def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        transformer_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None
        for model_ in models:
            if isinstance(model_, type(accelerator.unwrap_model(model))):
                if args.use_lora:
                    transformer_lora_layers_to_save = (
                        get_peft_model_state_dict(model_))
                else:
                    model_.save_pretrained(os.path.join(output_dir,
                        'transformer'))
            elif isinstance(model_, type(accelerator.unwrap_model(
                text_encoder))):
                if args.text_encoder_use_lora:
                    text_encoder_lora_layers_to_save = (
                        get_peft_model_state_dict(model_))
                else:
                    model_.save_pretrained(os.path.join(output_dir,
                        'text_encoder'))
            else:
                raise ValueError(f'unexpected save model: {model_.__class__}')
            weights.pop()
        if (transformer_lora_layers_to_save is not None or 
            text_encoder_lora_layers_to_save is not None):
            LoraLoaderMixin.save_lora_weights(output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save)
        if args.use_ema:
            ema.save_pretrained(os.path.join(output_dir, 'ema_model'))
