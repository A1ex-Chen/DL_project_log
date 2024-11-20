def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + '_' + str(idx)):
        idx += 1
    return idx
