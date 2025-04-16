def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D,
        CrossAttnUpBlock2D, UpBlock2D)):
        module.gradient_checkpointing = value
