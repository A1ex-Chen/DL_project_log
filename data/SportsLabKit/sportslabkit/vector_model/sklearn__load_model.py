def _load_model(self, path: str) ->None:
    """
        Load a scikit-learn pipeline model from disk using joblib.

        This method uses joblib to load a pre-trained scikit-learn pipeline from the specified file path.
        The loaded model is stored in the `model` attribute. A type check is performed to ensure
        that the loaded object is a scikit-learn pipeline.

        Args:
            path (str): The file path to the pre-trained scikit-learn pipeline.

        Raises:
            TypeError: If the loaded model is not a scikit-learn pipeline.
        """
    actual_path = fetch_or_cache_model(path)
    self.model = load(actual_path)
    if not isinstance(self.model, Pipeline):
        raise TypeError(
            f"Oops, you loaded something that's not a pipeline. Got a {type(self.model)} instead."
            )
