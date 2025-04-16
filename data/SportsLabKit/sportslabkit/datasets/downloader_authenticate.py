def authenticate(show_message: bool=True) ->Any:
    """Authenticate the Kaggle API key."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info('Authentication successful.')
    except OSError:
        logger.error(
            'Kaggle API key not found. Showing instructions to authenticate.')
        if show_message:
            return show_authenticate_message()
        return None
    return api
