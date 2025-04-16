def load_config(yaml_path: str) ->OmegaConf:
    """Load config from yaml file.

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        OmegaConf: Config object loaded from yaml file
    """
    assert os.path.exists(yaml_path)
    cfg = OmegaConf.load(yaml_path)
    cfg.outdir = cfg.outdir
    os.makedirs(cfg.outdir, exist_ok=True)
    return cfg
