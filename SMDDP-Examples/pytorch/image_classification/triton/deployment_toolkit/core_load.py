@abc.abstractmethod
def load(self, model_path: Union[str, Path], **kwargs) ->Model:
    """
        Loads and process model from file based on given set of args
        """
    pass
