def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, 'code')
    if os.path.exists(new_code_path):
        print(
            f'Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.'
            )
        return -1
    print(f'Copying codebase to {new_code_path}')
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log',
        'logs', 'wandb'))
    print('Done copying code.')
    return 1
