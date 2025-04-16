def load_model_hook(models, input_dir):
    unet_ = accelerator.unwrap_model(unet)
    lora_state_dict, _ = StableDiffusionXLPipeline.lora_state_dict(input_dir)
    unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in
        lora_state_dict.items() if k.startswith('unet.')}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict,
        adapter_name='default')
    if incompatible_keys is not None:
        unexpected_keys = getattr(incompatible_keys, 'unexpected_keys', None)
        if unexpected_keys:
            logger.warning(
                f'Loading adapter weights from state_dict led to unexpected keys not found in the model:  {unexpected_keys}. '
                )
    for _ in range(len(models)):
        models.pop()
    if args.mixed_precision == 'fp16':
        cast_training_params(unet_, dtype=torch.float32)
