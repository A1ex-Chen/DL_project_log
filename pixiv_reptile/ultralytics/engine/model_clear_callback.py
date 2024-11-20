def clear_callback(self, event: str) ->None:
    """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.

        Args:
            event (str): The name of the event for which to clear the callbacks.

        Raises:
            ValueError: If the event name is not recognized.
        """
    self.callbacks[event] = []
