def load_model_hook(models, input_dir):
    unet_ = None
    text_encoder_one_ = None
    text_encoder_two_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(unwrap_model(unet))):
            unet_ = model
        elif isinstance(model, type(unwrap_model(text_encoder_one))):
            text_encoder_one_ = model
        elif isinstance(model, type(unwrap_model(text_encoder_two))):
            text_encoder_two_ = model
        else:
            raise ValueError(f'unexpected save model: {model.__class__}')
    lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
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
    if args.train_text_encoder:
        _set_state_dict_into_text_encoder(lora_state_dict, prefix=
            'text_encoder.', text_encoder=text_encoder_one_)
        _set_state_dict_into_text_encoder(lora_state_dict, prefix=
            'text_encoder_2.', text_encoder=text_encoder_two_)
    if args.mixed_precision == 'fp16':
        models = [unet_]
        if args.train_text_encoder:
            models.extend([text_encoder_one_, text_encoder_two_])
        cast_training_params(models, dtype=torch.float32)
