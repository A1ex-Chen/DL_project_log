def to_dict(self):
    """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
    output = copy.deepcopy(self.__dict__)
    output['vision_config'] = self.vision_config.to_dict()
    output['text_config'] = self.text_config.to_dict()
    output['model_type'] = self.__class__.model_type
    output['cross_attn_every_n_layers'] = self.cross_attn_every_n_layers
    output['use_media_placement_augmentation'
        ] = self.use_media_placement_augmentation
    return output
