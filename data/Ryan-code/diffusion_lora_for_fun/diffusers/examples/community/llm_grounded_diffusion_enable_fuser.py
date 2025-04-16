def enable_fuser(self, enabled=True):
    for module in self.unet.modules():
        if isinstance(module, GatedSelfAttentionDense):
            module.enabled = enabled
