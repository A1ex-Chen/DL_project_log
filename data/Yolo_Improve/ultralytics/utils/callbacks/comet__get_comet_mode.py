def _get_comet_mode():
    """Returns the mode of comet set in the environment variables, defaults to 'online' if not set."""
    return os.getenv('COMET_MODE', 'online')
