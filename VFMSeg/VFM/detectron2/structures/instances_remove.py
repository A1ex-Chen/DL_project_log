def remove(self, name: str) ->None:
    """
        Remove the field called `name`.
        """
    del self._fields[name]
