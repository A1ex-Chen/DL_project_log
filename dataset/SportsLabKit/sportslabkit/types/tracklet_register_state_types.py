def register_state_types(self, names: list[str]) ->None:
    """Register a new state type.

        Args:
            name (str): Name of the new state type to be registered.
        """
    for name in names:
        self.register_state_type(name)
