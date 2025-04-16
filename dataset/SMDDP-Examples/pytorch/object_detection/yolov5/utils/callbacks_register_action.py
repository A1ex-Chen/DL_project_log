def register_action(self, hook, name='', callback=None):
    """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
    assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
    assert callable(callback), f"callback '{callback}' is not callable"
    self._callbacks[hook].append({'name': name, 'callback': callback})
