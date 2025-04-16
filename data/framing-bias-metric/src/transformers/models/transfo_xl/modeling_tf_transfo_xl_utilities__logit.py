@staticmethod
def _logit(x, W, b, proj=None):
    y = x
    if proj is not None:
        y = tf.einsum('ibd,ed->ibe', y, proj)
    return tf.einsum('ibd,nd->ibn', y, W) + b
