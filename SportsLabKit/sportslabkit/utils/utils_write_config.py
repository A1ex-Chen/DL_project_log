def write_config(yaml_path: str, cfg: OmegaConf) ->None:
    """Write config to yaml file.

    Args:
        yaml_path (str): Path to yaml file
        cfg (OmegaConf): Config object
    """
    assert os.path.exists(yaml_path)
    OmegaConf.save(cfg, yaml_path)
