def unfuse_lora(self):
    self.apply(self._unfuse_lora_apply)
