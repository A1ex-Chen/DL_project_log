def run(self, hook, *args, **kwargs):
    """
        Loop through the registered actions and fire all callbacks

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            kwargs: Keyword Arguments to receive from YOLOv5
        """
    assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
    for logger in self._callbacks[hook]:
        logger['callback'](*args, **kwargs)
