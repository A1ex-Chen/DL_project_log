def scaled_dot_product_attention(q, k, v, mask, attention_mask=None,
    head_mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(shape_list(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -10000.0
    if attention_mask is not None:
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
