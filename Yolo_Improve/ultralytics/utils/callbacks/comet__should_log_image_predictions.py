def _should_log_image_predictions():
    """Determines whether to log image predictions based on a specified environment variable."""
    return os.getenv('COMET_EVAL_LOG_IMAGE_PREDICTIONS', 'true').lower(
        ) == 'true'
