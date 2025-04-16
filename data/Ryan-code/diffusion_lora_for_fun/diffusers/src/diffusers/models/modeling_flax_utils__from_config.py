@classmethod
def _from_config(cls, config, **kwargs):
    """
        All context managers that the model should be initialized under go here.
        """
    return cls(config, **kwargs)
