def reset(self):
    super(AttentionStore, self).reset()
    self.step_store = self.get_empty_store()
    self.attention_store = {}
