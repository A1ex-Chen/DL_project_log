def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (Encoder, Decoder)):
        module.gradient_checkpointing = value
