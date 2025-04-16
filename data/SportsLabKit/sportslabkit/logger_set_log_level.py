def set_log_level(level: str) ->Any:
    """Set the logging level for the logger.

    Args:
        level (str): Logging level to set
    """
    level_filter.level = level
    os.environ['LOG_LEVEL'] = level
