def call(self, x):
    out = tf.reshape(x, (-1, 256, self.decoder_input_size, self.
        decoder_input_size))
    out = self.m3(self.m2(self.m1(out)))
    return self.bottle(out)
