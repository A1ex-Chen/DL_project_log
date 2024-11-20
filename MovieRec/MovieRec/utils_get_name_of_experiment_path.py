def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, experiment_description +
        '_' + str(date.today()))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + '_' + str(idx)
    return experiment_path
