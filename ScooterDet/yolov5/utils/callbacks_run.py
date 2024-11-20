def run(self, hook, *args, thread=False, **kwargs):
    """
        Loop through the registered actions and fire all callbacks on main thread

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            thread: (boolean) Run callbacks in daemon thread
            kwargs: Keyword Arguments to receive from YOLOv5
        """
    assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
    for logger in self._callbacks[hook]:
        if thread:
            threading.Thread(target=logger['callback'], args=args, kwargs=
                kwargs, daemon=True).start()
        else:
            logger['callback'](*args, **kwargs)
