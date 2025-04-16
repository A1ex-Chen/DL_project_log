@classmethod
def create_session(cls, identifier, args=None):
    """Class method to create an authenticated HUBTrainingSession or return None."""
    try:
        session = cls(identifier)
        if not session.client.authenticated:
            if identifier.startswith(f'{HUB_WEB_ROOT}/models/'):
                LOGGER.warning(
                    f"{PREFIX}WARNING ⚠️ Login to Ultralytics HUB with 'yolo hub login API_KEY'."
                    )
                exit()
            return None
        if args and not identifier.startswith(f'{HUB_WEB_ROOT}/models/'):
            session.create_model(args)
            assert session.model.id, 'HUB model not loaded correctly'
        return session
    except (PermissionError, ModuleNotFoundError, AssertionError):
        return None
