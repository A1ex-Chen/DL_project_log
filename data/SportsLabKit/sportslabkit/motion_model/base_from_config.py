@classmethod
def from_config(cls: type['BaseMotionModel'], config: dict
    ) ->'BaseMotionModel':
    """Initialize a motion model instance from a configuration dictionary.

        Args:
            config (Dict): The configuration dictionary containing the motion model's parameters.

        Returns:
            MotionModel: A new instance of the motion model initialized with the given configuration.
        """
    return cls(**config)
