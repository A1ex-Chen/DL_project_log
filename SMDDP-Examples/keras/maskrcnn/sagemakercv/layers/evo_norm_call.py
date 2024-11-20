@tf.function
def call(self, x):
    if self.nonlinear:
        num = x * tf.nn.sigmoid(self.v * x)
        return num / self._group_std(x) * self.gamma + self.beta
    else:
        return x * self.gamma + self.beta
