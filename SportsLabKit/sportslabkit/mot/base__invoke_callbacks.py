def _invoke_callbacks(self, method_name):
    """
        Invokes the appropriate methods on all callback objects.

        Args:
            method_name (str): The name of the method to invoke on the callback objects.
        """
    for callback in self.callbacks:
        method = getattr(callback, method_name, None)
        if method:
            method(self)
