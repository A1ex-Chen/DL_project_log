def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, LlavaMPTModel):
        module.gradient_checkpointing = value
