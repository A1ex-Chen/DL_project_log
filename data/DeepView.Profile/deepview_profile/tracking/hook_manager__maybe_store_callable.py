def _maybe_store_callable(self, module, prop, original_callable):
    """
        Store the original callable (to be able to restore it) only when it is
        the first time we are encountering the given callable.
        """
    if module not in self._original_callables:
        self._original_callables[module] = {}
    if prop in self._original_callables[module]:
        return
    self._original_callables[module][prop] = original_callable
