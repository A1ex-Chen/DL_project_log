@classmethod
def from_json_file(cls, json_file: Union[str, os.PathLike]
    ) ->'PretrainedConfig':
    """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
    config_dict = cls._dict_from_json_file(json_file)
    return cls(**config_dict)
