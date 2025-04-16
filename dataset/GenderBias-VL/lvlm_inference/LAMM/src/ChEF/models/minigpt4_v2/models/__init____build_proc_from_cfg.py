def _build_proc_from_cfg(cfg):
    return registry.get_processor_class(cfg.name).from_config(cfg
        ) if cfg is not None else BaseProcessor()
