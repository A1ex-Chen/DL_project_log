@staticmethod
def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
    pos_emb = pos_emb[:, None, :]
    if bsz is not None:
        pos_emb = tf.tile(pos_emb, [1, bsz, 1])
    return pos_emb
