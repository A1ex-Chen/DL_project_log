def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (T5Attention, T5Stack)):
        module.gradient_checkpointing = value
