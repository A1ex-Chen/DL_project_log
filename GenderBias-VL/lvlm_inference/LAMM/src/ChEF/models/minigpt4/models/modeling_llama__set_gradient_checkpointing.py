def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, LlamaModel):
        module.gradient_checkpointing = value
