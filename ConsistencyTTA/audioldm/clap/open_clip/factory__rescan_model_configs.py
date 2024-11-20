def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = '.json',
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))
    for cf in config_files:
        if os.path.basename(cf)[0] == '.':
            continue
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'audio_cfg',
                'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=
        lambda x: _natural_key(x[0]))}
