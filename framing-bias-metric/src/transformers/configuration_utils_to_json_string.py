def to_json_string(self, use_diff: bool=True) ->str:
    """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
    if use_diff is True:
        config_dict = self.to_diff_dict()
    else:
        config_dict = self.to_dict()
    return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'
