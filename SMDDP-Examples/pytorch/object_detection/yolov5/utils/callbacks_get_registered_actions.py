def get_registered_actions(self, hook=None):
    """"
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
    return self._callbacks[hook] if hook else self._callbacks
