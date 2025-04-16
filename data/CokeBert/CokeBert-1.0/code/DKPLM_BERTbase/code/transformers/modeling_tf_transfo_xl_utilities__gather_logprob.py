@staticmethod
def _gather_logprob(logprob, target):
    lp_size = tf.shape(logprob)
    r = tf.range(lp_size[0])
    idx = tf.stack([r, target], 1)
    return tf.gather_nd(logprob, idx)
