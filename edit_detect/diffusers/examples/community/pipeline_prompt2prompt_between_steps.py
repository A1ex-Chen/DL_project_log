def between_steps(self):
    if len(self.attention_store) == 0:
        self.attention_store = self.step_store
    else:
        for key in self.attention_store:
            for i in range(len(self.attention_store[key])):
                self.attention_store[key][i] += self.step_store[key][i]
    self.step_store = self.get_empty_store()
