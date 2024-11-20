def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (CrossAttnDownBlockFlat, DownBlockFlat,
        CrossAttnUpBlockFlat, UpBlockFlat)):
        module.gradient_checkpointing = value
