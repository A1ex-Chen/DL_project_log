def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re
    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub('(\\d+)([nslmx])6(.+)?$', '\\1\\2-p6\\3', path.stem)
        LOGGER.warning(
            f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.'
            )
        path = path.with_name(new_stem + path.suffix)
    unified_path = re.sub('(\\d+)([nslmx])(.+)?$', '\\1\\3', str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d
