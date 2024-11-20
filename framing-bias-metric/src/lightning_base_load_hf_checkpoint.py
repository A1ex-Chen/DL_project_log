def load_hf_checkpoint(self, *args, **kwargs):
    self.model = self.model_type.from_pretrained(*args, **kwargs)
