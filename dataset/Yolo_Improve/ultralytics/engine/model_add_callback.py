def add_callback(self, event: str, func) ->None:
    """
        Adds a callback function for a specified event.

        This method allows the user to register a custom callback function that is triggered on a specific event during
        model training or inference.

        Args:
            event (str): The name of the event to attach the callback to.
            func (callable): The callback function to be registered.

        Raises:
            ValueError: If the event name is not recognized.
        """
    self.callbacks[event].append(func)
