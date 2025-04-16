def to_dict(self) ->Dict[str, Any]:
    """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
    output = copy.deepcopy(self.__dict__)
    if hasattr(self.__class__, 'model_type'):
        output['model_type'] = self.__class__.model_type
    return output
