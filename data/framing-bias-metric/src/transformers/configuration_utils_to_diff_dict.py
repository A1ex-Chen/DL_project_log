def to_diff_dict(self) ->Dict[str, Any]:
    """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
    config_dict = self.to_dict()
    default_config_dict = PretrainedConfig().to_dict()
    class_config_dict = self.__class__().to_dict(
        ) if not self.is_composition else {}
    serializable_config_dict = {}
    for key, value in config_dict.items():
        if key not in default_config_dict or value != default_config_dict[key
            ] or key in class_config_dict and value != class_config_dict[key]:
            serializable_config_dict[key] = value
    return serializable_config_dict
