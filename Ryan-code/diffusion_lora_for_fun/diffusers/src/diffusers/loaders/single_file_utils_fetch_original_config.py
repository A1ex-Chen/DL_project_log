def fetch_original_config(original_config_file, local_files_only=False):
    if os.path.isfile(original_config_file):
        with open(original_config_file, 'r') as fp:
            original_config_file = fp.read()
    elif is_valid_url(original_config_file):
        if local_files_only:
            raise ValueError(
                '`local_files_only` is set to True, but a URL was provided as `original_config_file`. Please provide a valid local file path.'
                )
        original_config_file = BytesIO(requests.get(original_config_file).
            content)
    else:
        raise ValueError(
            'Invalid `original_config_file` provided. Please set it to a valid file path or URL.'
            )
    original_config = yaml.safe_load(original_config_file)
    return original_config
