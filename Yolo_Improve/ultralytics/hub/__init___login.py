def login(api_key: str=None, save=True) ->bool:
    """
    Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY
    environment variable if successfully authenticated.

    Args:
        api_key (str, optional): API key to use for authentication.
            If not provided, it will be retrieved from SETTINGS or HUB_API_KEY environment variable.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    checks.check_requirements('hub-sdk>=0.0.8')
    from hub_sdk import HUBClient
    api_key_url = f'{HUB_WEB_ROOT}/settings?tab=api+keys'
    saved_key = SETTINGS.get('api_key')
    active_key = api_key or saved_key
    credentials = {'api_key': active_key
        } if active_key and active_key != '' else None
    client = HUBClient(credentials)
    if client.authenticated:
        if save and client.api_key != saved_key:
            SETTINGS.update({'api_key': client.api_key})
        log_message = ('New authentication successful ✅' if client.api_key ==
            api_key or not credentials else 'Authenticated ✅')
        LOGGER.info(f'{PREFIX}{log_message}')
        return True
    else:
        LOGGER.info(
            f"{PREFIX}Get API key from {api_key_url} and then run 'yolo hub login API_KEY'"
            )
        return False
