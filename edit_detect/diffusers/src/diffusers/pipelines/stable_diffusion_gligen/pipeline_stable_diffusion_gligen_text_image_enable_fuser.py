def enable_fuser(self, enabled=True):
    for module in self.unet.modules():
        if type(module) is GatedSelfAttentionDense:
            module.enabled = enabled
