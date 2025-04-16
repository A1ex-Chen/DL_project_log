def __init__(self, api_key='', verbose=False):
    """
        Initialize the Auth class with an optional API key.

        Args:
            api_key (str, optional): May be an API key or a combination API key and model ID, i.e. key_id
        """
    api_key = api_key.split('_')[0]
    self.api_key = api_key or SETTINGS.get('api_key', '')
    if self.api_key:
        if self.api_key == SETTINGS.get('api_key'):
            if verbose:
                LOGGER.info(f'{PREFIX}Authenticated ✅')
            return
        else:
            success = self.authenticate()
    elif IS_COLAB:
        success = self.auth_with_cookies()
    else:
        success = self.request_api_key()
    if success:
        SETTINGS.update({'api_key': self.api_key})
        if verbose:
            LOGGER.info(f'{PREFIX}New authentication successful ✅')
    elif verbose:
        LOGGER.info(
            f"{PREFIX}Get API key from {API_KEY_URL} and then run 'yolo hub login API_KEY'"
            )
