def load_opt_from_config_files(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    load_config_dict_to_opt(opt, config_dict)
    return opt
