def register_observation_types(self, names: list[str]) ->None:
    """Register a new observation type.

        Args:
            name (str): Name of the new observation type to be registered.
        """
    for name in names:
        self.register_observation_type(name)
