def to_json_string(self) ->str:
    """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
    config_dict = self._internal_dict if hasattr(self, '_internal_dict') else {
        }
    config_dict['_class_name'] = self.__class__.__name__
    config_dict['_diffusers_version'] = __version__

    def to_json_saveable(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, PosixPath):
            value = str(value)
        return value
    config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
    config_dict.pop('_ignore_files', None)
    config_dict.pop('_use_default_values', None)
    return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'
