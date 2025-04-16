def _set_gradient_checkpointing(self, module, value: bool=False) ->None:
    if isinstance(module, (EncoderTiny, DecoderTiny)):
        module.gradient_checkpointing = value
