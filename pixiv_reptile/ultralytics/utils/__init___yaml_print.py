def yaml_print(yaml_file: Union[str, Path, dict]) ->None:
    """
    Pretty prints a YAML file or a YAML-formatted dictionary.

    Args:
        yaml_file: The file path of the YAML file or a YAML-formatted dictionary.

    Returns:
        (None)
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)
        ) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=
        float('inf'))
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")
