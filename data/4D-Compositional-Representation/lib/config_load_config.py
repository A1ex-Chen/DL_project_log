def load_config(path, default_path=None):
    """ Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)
    inherit_from = cfg_special.get('inherit_from')
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()
    update_recursive(cfg, cfg_special)
    return cfg
