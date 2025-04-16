def load(self, path: str) ->None:
    """Load the model from disk.

        Args:
            path (str): The path to the model file.
        """
    self._load_model(path)
    self._post_load_check()
