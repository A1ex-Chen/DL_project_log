def to_dict(self):
    """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
    output = copy.deepcopy(self.__dict__)
    output['encoder'] = self.encoder.to_dict()
    output['decoder'] = self.decoder.to_dict()
    output['model_type'] = self.__class__.model_type
    return output
