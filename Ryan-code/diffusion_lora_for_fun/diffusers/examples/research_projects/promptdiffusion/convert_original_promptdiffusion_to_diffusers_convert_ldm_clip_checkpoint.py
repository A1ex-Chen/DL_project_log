def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False,
    text_encoder=None):
    if text_encoder is None:
        config_name = 'openai/clip-vit-large-patch14'
        try:
            config = CLIPTextConfig.from_pretrained(config_name,
                local_files_only=local_files_only)
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
                )
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            text_model = CLIPTextModel(config)
    else:
        text_model = text_encoder
    keys = list(checkpoint.keys())
    text_model_dict = {}
    remove_prefixes = ['cond_stage_model.transformer',
        'conditioner.embedders.0.transformer']
    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + '.'):]] = checkpoint[key]
    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(text_model, param_name, 'cpu',
                value=param)
    else:
        if not (hasattr(text_model, 'embeddings') and hasattr(text_model.
            embeddings.position_ids)):
            text_model_dict.pop('text_model.embeddings.position_ids', None)
        text_model.load_state_dict(text_model_dict)
    return text_model
