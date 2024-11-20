def _set_gradient_checkpointing(self, module, value: bool=False) ->None:
    if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
        module.gradient_checkpointing = value
