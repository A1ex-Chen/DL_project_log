def _get_comet_model_name():
    """Returns the model name for Comet from the environment variable 'COMET_MODEL_NAME' or defaults to 'YOLOv8'."""
    return os.getenv('COMET_MODEL_NAME', 'YOLOv8')
