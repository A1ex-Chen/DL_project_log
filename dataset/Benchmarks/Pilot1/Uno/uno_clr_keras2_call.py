def call(self, x, mask=None):
    if 0.0 < self.rate < 1.0:
        noise_shape = self._get_noise_shape(x)
        x = K.dropout(x, self.rate, noise_shape)
    return x
