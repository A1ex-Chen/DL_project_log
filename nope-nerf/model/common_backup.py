def backup(out_dir, config):
    backup_path = os.path.join(out_dir, 'backup')
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    shutil.copyfile(config, os.path.join(backup_path, 'config.yaml'))
    shutil.copy('train.py', backup_path)
    shutil.copy('./configs/default.yaml', backup_path)
    base_dirs = ['./model', './dataloading']
    for base_dir in base_dirs:
        files_ = [f for f in os.listdir(base_dir) if os.path.isfile(os.path
            .join(base_dir, f))]
        backup_subpath = os.path.join(backup_path, base_dir[2:])
        if not os.path.exists(backup_subpath):
            os.makedirs(backup_subpath)
        for file in files_:
            shutil.copy(os.path.join(base_dir, file), backup_subpath)
