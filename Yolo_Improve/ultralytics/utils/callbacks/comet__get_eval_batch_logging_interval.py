def _get_eval_batch_logging_interval():
    """Get the evaluation batch logging interval from environment variable or use default value 1."""
    return int(os.getenv('COMET_EVAL_BATCH_LOGGING_INTERVAL', 1))
