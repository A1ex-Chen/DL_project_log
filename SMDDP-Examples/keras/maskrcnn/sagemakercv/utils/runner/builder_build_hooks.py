def build_hooks(cfg):
    return [HOOKS[hook](cfg) for hook in cfg.HOOKS]
