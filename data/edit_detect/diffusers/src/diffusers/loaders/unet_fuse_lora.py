def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    self.lora_scale = lora_scale
    self._safe_fusing = safe_fusing
    self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))
