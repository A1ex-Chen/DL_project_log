def to_dict(self):
    """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
    output = copy.deepcopy(self.__dict__)
    output['question_encoder'] = self.question_encoder.to_dict()
    output['generator'] = self.generator.to_dict()
    output['model_type'] = self.__class__.model_type
    return output
