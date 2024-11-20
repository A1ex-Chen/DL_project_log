def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff:
    bool=True):
    """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
    with open(json_file_path, 'w', encoding='utf-8') as writer:
        writer.write(self.to_json_string(use_diff=use_diff))
