def between_steps(self):
    self.attention_store = self.step_store
    self.step_store = self.get_empty_store()
