def attach_hook(self, module, prop, hook_creator):
    target = getattr(module, prop)
    self._maybe_store_callable(module, prop, target)
    setattr(module, prop, hook_creator(target))
