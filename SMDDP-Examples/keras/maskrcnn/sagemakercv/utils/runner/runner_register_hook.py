def register_hook(self, hook, priority='NORMAL'):
    """
        Register a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
    assert isinstance(hook, Hook)
    if hasattr(hook, 'priority'):
        raise ValueError('"priority" is a reserved attribute for hooks')
    priority = get_priority(priority)
    hook.priority = priority
    inserted = False
    for i in range(len(self._hooks) - 1, -1, -1):
        if priority >= self._hooks[i].priority:
            self._hooks.insert(i + 1, hook)
            inserted = True
            break
    if not inserted:
        self._hooks.insert(0, hook)
