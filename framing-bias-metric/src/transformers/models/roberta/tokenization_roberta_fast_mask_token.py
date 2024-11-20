@mask_token.setter
def mask_token(self, value):
    """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on Roberta.
        """
    value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value,
        str) else value
    self._mask_token = value
