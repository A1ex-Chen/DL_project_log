def remove_hooks(self):
    for module, callable_pairs in self._original_callables.items():
        for prop, original_callable in callable_pairs.items():
            setattr(module, prop, original_callable)
    self._original_callables.clear()
