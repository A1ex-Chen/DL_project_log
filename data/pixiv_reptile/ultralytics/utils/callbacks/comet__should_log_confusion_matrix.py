def _should_log_confusion_matrix():
    """Determines if the confusion matrix should be logged based on the environment variable settings."""
    return os.getenv('COMET_EVAL_LOG_CONFUSION_MATRIX', 'false').lower(
        ) == 'true'
