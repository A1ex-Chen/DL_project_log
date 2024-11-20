def _set_gradient_checkpointing(self, module, value: bool=False) ->None:
    if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D,
        CrossAttnUpBlock3D, UpBlock3D)):
        module.gradient_checkpointing = value
