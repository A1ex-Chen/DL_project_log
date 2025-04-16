def _set_gradient_checkpointing(self, module, value: bool=False) ->None:
    if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion,
        CrossAttnUpBlockMotion, UpBlockMotion)):
        module.gradient_checkpointing = value
