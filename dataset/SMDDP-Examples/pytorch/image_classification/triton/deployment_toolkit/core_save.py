@abc.abstractmethod
def save(self, model: Model, model_path: Union[str, Path]) ->None:
    """
        Save model to file
        """
    pass
