def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return registry.get_processor_class(cfg.name).from_config(cfg
            ) if cfg is not None else BaseProcessor()
    vis_processors = dict()
    txt_processors = dict()
    vis_proc_cfg = config.get('vis_processor')
    txt_proc_cfg = config.get('text_processor')
    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get('train')
        vis_eval_cfg = vis_proc_cfg.get('eval')
    else:
        vis_train_cfg = None
        vis_eval_cfg = None
    vis_processors['train'] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors['eval'] = _build_proc_from_cfg(vis_eval_cfg)
    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get('train')
        txt_eval_cfg = txt_proc_cfg.get('eval')
    else:
        txt_train_cfg = None
        txt_eval_cfg = None
    txt_processors['train'] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors['eval'] = _build_proc_from_cfg(txt_eval_cfg)
    return vis_processors, txt_processors
