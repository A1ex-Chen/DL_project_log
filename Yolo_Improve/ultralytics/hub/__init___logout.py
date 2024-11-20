def logout():
    """
    Log out of Ultralytics HUB by removing the API key from the settings file. To log in again, use 'yolo hub login'.

    Example:
        ```python
        from ultralytics import hub

        hub.logout()
        ```
    """
    SETTINGS['api_key'] = ''
    SETTINGS.save()
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo hub login'."
        )
