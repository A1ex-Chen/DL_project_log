def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-
    float('Inf'), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits_shape = shape_list(logits)
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][...,
            -1, None]
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
            filter_value)
    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction='DESCENDING')
        sorted_logits = tf.gather(logits, sorted_indices, axis=-1, batch_dims=1
            )
        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis
            =-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove = tf.concat([tf.zeros_like(
                sorted_indices_to_remove[:, :min_tokens_to_keep]),
                sorted_indices_to_remove[:, min_tokens_to_keep:]], -1)
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1
            )
        sorted_indices_to_remove = tf.concat([tf.zeros_like(
            sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, 1
            :]], -1)
        indices_to_remove = scatter_values_on_batch_indices(
            sorted_indices_to_remove, sorted_indices)
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
            filter_value)
    return logits
