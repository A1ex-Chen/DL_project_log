def check_class_names(names):
    """Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices {min(names.keys())}-{max(names.keys())} defined in your dataset YAML.'
                )
        if isinstance(names[0], str) and names[0].startswith('n0'):
            map = yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']
            names = {k: map[v] for k, v in names.items()}
    return names
