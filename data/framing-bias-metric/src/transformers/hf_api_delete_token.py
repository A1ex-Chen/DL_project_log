@classmethod
def delete_token(cls):
    """
        Delete token. Do not fail if token does not exist.
        """
    try:
        os.remove(cls.path_token)
    except FileNotFoundError:
        pass
