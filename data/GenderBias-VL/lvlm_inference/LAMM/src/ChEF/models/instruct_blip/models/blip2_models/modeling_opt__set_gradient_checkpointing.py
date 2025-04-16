def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, OPTDecoder):
        module.gradient_checkpointing = value
