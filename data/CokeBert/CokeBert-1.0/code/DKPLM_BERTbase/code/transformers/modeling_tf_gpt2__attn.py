def _attn(self, inputs, training=False):
    q, k, v, attention_mask, head_mask = inputs
    w = tf.matmul(q, k, transpose_b=True)
    if self.scale:
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
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
    if self.output_attentions:
        outputs.append(w)
    return outputs
