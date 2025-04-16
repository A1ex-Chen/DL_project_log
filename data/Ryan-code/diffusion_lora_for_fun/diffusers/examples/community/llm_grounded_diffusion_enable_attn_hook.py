def enable_attn_hook(self, enabled=True):
    for module in self.unet.attn_processors.values():
        if isinstance(module, AttnProcessorWithHook):
            module.enabled = enabled
