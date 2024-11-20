@classmethod
def save_token(cls, token):
    """
        Save token, creating folder as needed.
        """
    os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
    with open(cls.path_token, 'w+') as f:
        f.write(token)
