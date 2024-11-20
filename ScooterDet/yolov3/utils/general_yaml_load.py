def yaml_load(file='data.yaml'):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)
