def _get_default_logging_level():
    """
    If DIFFUSERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv('DIFFUSERS_VERBOSITY', None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option DIFFUSERS_VERBOSITY={env_level_str}, has to be one of: {', '.join(log_levels.keys())}"
                )
    return _default_log_level
