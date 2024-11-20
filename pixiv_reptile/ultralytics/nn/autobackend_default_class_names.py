def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))['names']
    return {i: f'class{i}' for i in range(999)}
