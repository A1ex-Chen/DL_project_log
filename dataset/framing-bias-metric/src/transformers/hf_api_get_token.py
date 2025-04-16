@classmethod
def get_token(cls):
    """
        Get token or None if not existent.
        """
    try:
        with open(cls.path_token, 'r') as f:
            return f.read()
    except FileNotFoundError:
        pass
