def _set_gradient_checkpointing(self, module, value=False):
    if hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing = value
