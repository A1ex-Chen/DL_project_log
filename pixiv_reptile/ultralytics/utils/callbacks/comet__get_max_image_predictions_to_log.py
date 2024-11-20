def _get_max_image_predictions_to_log():
    """Get the maximum number of image predictions to log from the environment variables."""
    return int(os.getenv('COMET_MAX_IMAGE_PREDICTIONS', 100))
