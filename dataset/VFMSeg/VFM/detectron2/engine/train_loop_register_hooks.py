def register_hooks(self, hooks: List[Optional[HookBase]]) ->None:
    """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
    hooks = [h for h in hooks if h is not None]
    for h in hooks:
        assert isinstance(h, HookBase)
        h.trainer = weakref.proxy(self)
    self._hooks.extend(hooks)
