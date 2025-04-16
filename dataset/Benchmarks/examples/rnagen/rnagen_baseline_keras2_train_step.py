def train_step(self, data):
    return self.custom_step(data, train=True)
