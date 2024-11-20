def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin':
            'Library/Application Support'}
        path = Path.home() / cfg.get(platform.system(), '')
        path = (path if is_writeable(path) else Path('/tmp')) / dir
    path.mkdir(exist_ok=True)
    return path
