def has(self, name: str) ->bool:
    """
        Returns:
            bool: whether the field called `name` exists.
        """
    return name in self._fields
