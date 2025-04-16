def patch_config(config):
    patch_dict = {'use_mm_proj': True, 'mm_vision_tower':
        'openai/clip-vit-large-patch14', 'mm_hidden_size': 1024}
    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, 'mm_vision_tower'):
        print(
            f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.'
            )
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)
