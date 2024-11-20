def to_tuple(self) ->Tuple[Any]:
    """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
    return tuple(self[k] for k in self.keys())
