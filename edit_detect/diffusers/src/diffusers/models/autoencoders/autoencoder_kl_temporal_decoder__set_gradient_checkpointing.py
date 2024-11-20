def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (Encoder, TemporalDecoder)):
        module.gradient_checkpointing = value
