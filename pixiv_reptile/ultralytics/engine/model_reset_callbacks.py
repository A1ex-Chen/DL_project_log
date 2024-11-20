def reset_callbacks(self) ->None:
    """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        added previously.
        """
    for event in callbacks.default_callbacks.keys():
        self.callbacks[event] = [callbacks.default_callbacks[event][0]]
