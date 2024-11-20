def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, CLIPEncoder):
        module.gradient_checkpointing = value
