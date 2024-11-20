def to_json_file(self, json_file_path: Union[str, os.PathLike]):
    """
        Save the configuration instance's parameters to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file to save a configuration instance's parameters.
        """
    with open(json_file_path, 'w', encoding='utf-8') as writer:
        writer.write(self.to_json_string())
