def attach_hooks_on_module_using(self, module, using_module, predicate,
    hook_creator):
    """
        Attach hooks onto functions in the provided module. Use the
        `using_module` to discover the existing functions.
        """
    for prop in dir(using_module):
        if not predicate(getattr(module, prop)):
            continue
        self.attach_hook(module, prop, hook_creator)
