def check_comet_weights(opt):
    """
    Downloads model weights from Comet and updates the weights path to point to saved weights location.

    Args:
        opt (argparse.Namespace): Command Line arguments passed
            to YOLOv3 training script

    Returns:
        None/bool: Return True if weights are successfully downloaded
            else return None
    """
    if comet_ml is None:
        return
    if isinstance(opt.weights, str) and opt.weights.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.weights)
        experiment_path = f'{resource.netloc}{resource.path}'
        experiment = api.get(experiment_path)
        download_model_checkpoint(opt, experiment)
        return True
    return None
