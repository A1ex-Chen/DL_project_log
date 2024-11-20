def create_experiment_export_folder(args):
    experiment_dir, experiment_description = (args.experiment_dir, args.
        experiment_description)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir,
        experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path
