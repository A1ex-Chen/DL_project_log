def _attn(self, q, k, v, attention_mask, head_mask, output_attentions,
    training=False):
    w = tf.matmul(q, k, transpose_b=True)
    if self.scale:
        dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)
        w = w / tf.math.sqrt(dk)
    _, _, nd, ns = shape_list(w)
    b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    w = w * b - 10000.0 * (1 - b)
    if attention_mask is not None:
        w = w + attention_mask
    w = tf.nn.softmax(w, axis=-1)
    w = self.attn_dropout(w, training=training)
    if head_mask is not None:
        w = w * head_mask
    outputs = [tf.matmul(w, v)]
    if output_attentions:
        outputs.append(w)
    return outputs
