def _legacy_load_clip_tokenizer(cls, checkpoint, config=None,
    local_files_only=False):
    if config:
        config = {'pretrained_model_name_or_path': config}
    else:
        config = fetch_diffusers_config(checkpoint)
    if is_clip_model(checkpoint) or is_clip_sdxl_model(checkpoint):
        clip_config = 'openai/clip-vit-large-patch14'
        config['pretrained_model_name_or_path'] = clip_config
        subfolder = ''
    elif is_open_clip_model(checkpoint):
        clip_config = 'stabilityai/stable-diffusion-2'
        config['pretrained_model_name_or_path'] = clip_config
        subfolder = 'tokenizer'
    else:
        clip_config = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        config['pretrained_model_name_or_path'] = clip_config
        subfolder = ''
    tokenizer = cls.from_pretrained(**config, subfolder=subfolder,
        local_files_only=local_files_only)
    return tokenizer
