@property
def config(self) ->Dict[str, Any]:
    """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
    return self._internal_dict
