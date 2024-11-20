def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)
