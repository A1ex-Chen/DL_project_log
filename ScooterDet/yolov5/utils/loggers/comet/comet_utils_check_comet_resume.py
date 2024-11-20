def check_comet_resume(opt):
    """Restores run parameters to its original state based on the model checkpoint
    and logged Experiment parameters.

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv5 training script

    Returns:
        None/bool: Return True if the run is restored successfully
            else return None
    """
    if comet_ml is None:
        return
    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            api = comet_ml.API()
            resource = urlparse(opt.resume)
            experiment_path = f'{resource.netloc}{resource.path}'
            experiment = api.get(experiment_path)
            set_opt_parameters(opt, experiment)
            download_model_checkpoint(opt, experiment)
            return True
    return None
