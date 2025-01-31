def safe_update(cfg, key, value):
    parts = key.split('.')
    for idx in range(1, len(parts)):
        prefix = '.'.join(parts[:idx])
        v = OmegaConf.select(cfg, prefix, default=None)
        if v is None:
            break
        if not OmegaConf.is_config(v):
            raise KeyError(
                f'Trying to update key {key}, but {prefix} is not a config, but has type {type(v)}.'
                )
    OmegaConf.update(cfg, key, value, merge=True)
