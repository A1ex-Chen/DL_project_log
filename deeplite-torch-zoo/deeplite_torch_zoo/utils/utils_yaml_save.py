def yaml_save(file='data.yaml', data={}):
    with open(file, 'w', encoding='utf-8') as f:
        yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v) for k, v in
            data.items()}, f, sort_keys=False)
