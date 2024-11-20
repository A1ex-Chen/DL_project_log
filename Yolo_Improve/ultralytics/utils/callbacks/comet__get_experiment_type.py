def _get_experiment_type(mode, project_name):
    """Return an experiment based on mode and project name."""
    if mode == 'offline':
        return comet_ml.OfflineExperiment(project_name=project_name)
    return comet_ml.Experiment(project_name=project_name)
