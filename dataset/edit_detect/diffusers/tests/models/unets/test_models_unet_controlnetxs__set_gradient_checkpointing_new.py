def _set_gradient_checkpointing_new(self, module, value=False):
    if hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing = value
        modules_with_gc_enabled[module.__class__.__name__] = True
