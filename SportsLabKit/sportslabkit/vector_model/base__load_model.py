@abstractmethod
def _load_model(self, path: str) ->None:
    """User Defined model loading logic. Must be overridden by subclasses.

        Args:
            path (str): The path to the model file.
        """
    raise NotImplementedError(
        'The _load_model method must be implemented by subclasses.')
