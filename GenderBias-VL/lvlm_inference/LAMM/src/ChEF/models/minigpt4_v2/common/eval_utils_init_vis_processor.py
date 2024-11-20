def init_vis_processor(args):
    print('Initialization Model')
    cfg = Config(args)
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
    print('Initialization Finished')
    return vis_processor
