def step(self):
    if self.i < len(self.drop_values):
        self.dropblock.drop_prob = self.drop_values[self.i]
    self.i += 1
