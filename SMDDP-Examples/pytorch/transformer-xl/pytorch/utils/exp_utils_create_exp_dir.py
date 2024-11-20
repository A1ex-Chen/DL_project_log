def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        return
    os.makedirs(dir_path, exist_ok=True)
    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(
                script))
            shutil.copyfile(script, dst_file)
