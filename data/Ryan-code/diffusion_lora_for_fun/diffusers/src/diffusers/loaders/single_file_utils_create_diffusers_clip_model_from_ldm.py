def create_diffusers_clip_model_from_ldm(cls, checkpoint, subfolder='',
    config=None, torch_dtype=None, local_files_only=None, is_legacy_loading
    =False):
    if config:
        config = {'pretrained_model_name_or_path': config}
    else:
        config = fetch_diffusers_config(checkpoint)
    if is_legacy_loading:
        logger.warning(
            'Detected legacy CLIP loading behavior. Please run `from_single_file` with `local_files_only=False once to update the local cache directory with the necessary CLIP model config files. Attempting to load CLIP model from legacy cache directory.'
            )
        if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
            clip_config = 'openai/clip-vit-large-patch14'
            config['pretrained_model_name_or_path'] = clip_config
            subfolder = ''
        elif is_open_clip_model(checkpoint):
            clip_config = 'stabilityai/stable-diffusion-2'
            config['pretrained_model_name_or_path'] = clip_config
            subfolder = 'text_encoder'
        else:
            clip_config = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
            config['pretrained_model_name_or_path'] = clip_config
            subfolder = ''
    model_config = cls.config_class.from_pretrained(**config, subfolder=
        subfolder, local_files_only=local_files_only)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        model = cls(model_config)
    position_embedding_dim = (model.text_model.embeddings.
        position_embedding.weight.shape[-1])
    if is_clip_model(checkpoint):
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)
    elif is_clip_sdxl_model(checkpoint) and checkpoint[CHECKPOINT_KEY_NAMES
        ['clip_sdxl']].shape[-1] == position_embedding_dim:
        diffusers_format_checkpoint = convert_ldm_clip_checkpoint(checkpoint)
    elif is_open_clip_model(checkpoint):
        prefix = 'cond_stage_model.model.'
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model,
            checkpoint, prefix=prefix)
    elif is_open_clip_sdxl_model(checkpoint) and checkpoint[
        CHECKPOINT_KEY_NAMES['open_clip_sdxl']].shape[-1
        ] == position_embedding_dim:
        prefix = 'conditioner.embedders.1.model.'
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model,
            checkpoint, prefix=prefix)
    elif is_open_clip_sdxl_refiner_model(checkpoint):
        prefix = 'conditioner.embedders.0.model.'
        diffusers_format_checkpoint = convert_open_clip_checkpoint(model,
            checkpoint, prefix=prefix)
    else:
        raise ValueError(
            'The provided checkpoint does not seem to contain a valid CLIP model.'
            )
    if is_accelerate_available():
        unexpected_keys = load_model_dict_into_meta(model,
            diffusers_format_checkpoint, dtype=torch_dtype)
        if model._keys_to_ignore_on_load_unexpected is not None:
            for pat in model._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(
                    pat, k) is None]
        if len(unexpected_keys) > 0:
            logger.warning(
                f"""Some weights of the model checkpoint were not used when initializing {cls.__name__}: 
 {[', '.join(unexpected_keys)]}"""
                )
    else:
        model.load_state_dict(diffusers_format_checkpoint)
    if torch_dtype is not None:
        model.to(torch_dtype)
    model.eval()
    return model
