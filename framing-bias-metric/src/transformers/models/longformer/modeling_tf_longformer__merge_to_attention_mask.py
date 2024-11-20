@staticmethod
def _merge_to_attention_mask(attention_mask: tf.Tensor,
    global_attention_mask: tf.Tensor):
    if attention_mask is not None:
        attention_mask = attention_mask * (global_attention_mask + 1)
    else:
        attention_mask = global_attention_mask + 1
    return attention_mask
