def predict(self, samples):
    output = self.forward(samples, is_train=False)
    return output
