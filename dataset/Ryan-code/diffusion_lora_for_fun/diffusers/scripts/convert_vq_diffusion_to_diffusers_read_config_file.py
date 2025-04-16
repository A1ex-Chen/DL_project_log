def read_config_file(filename):
    with open(filename) as f:
        original_config = yaml.load(f, FullLoader)
    return original_config
