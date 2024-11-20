def process_microbatch(microbatch_loss):
    """Compute clipped grads for one microbatch."""
    microbatch_loss = tf.reduce_mean(input_tensor=microbatch_loss)
    grads, _ = zip(*super(DPOptimizerClass, self).compute_gradients(
        microbatch_loss, var_list, gate_gradients, aggregation_method,
        colocate_gradients_with_ops, grad_loss))
    grads_list = [(g if g is not None else tf.zeros_like(v)) for g, v in
        zip(list(grads), var_list)]
    grads_flat = tf.nest.flatten(grads_list)
    squared_l2_norms = [tf.reduce_sum(input_tensor=tf.square(g)) for g in
        grads_flat]
    global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
    div = tf.maximum(global_norm / self._l2_norm_clip, 1.0)
    clipped_flat = [(g / div) for g in grads_flat]
    clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)
    return clipped_grads
