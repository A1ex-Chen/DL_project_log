def attach_hooks_on_module(self, module, predicate, hook_creator):
    self.attach_hooks_on_module_using(module, module, predicate, hook_creator)
