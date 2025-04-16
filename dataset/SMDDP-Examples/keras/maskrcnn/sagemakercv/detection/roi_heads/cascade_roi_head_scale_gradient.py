@tf.custom_gradient
def scale_gradient(self, x):
    return x, lambda dy: dy * (1.0 / self.num_stages)
