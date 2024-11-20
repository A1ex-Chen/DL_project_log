def compute_gradients(self, loss, var_list, gate_gradients=GATE_OP,
    aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=
    None, gradient_tape=None):
    """DP-SGD version of base class method."""
    self._was_compute_gradients_called = True
    if callable(loss):
        raise NotImplementedError('Vectorized optimizer unavailable for TF2.')
    else:
        if gradient_tape:
            raise ValueError('When in graph mode, a tape should not be passed.'
                )
        batch_size = tf.shape(input=loss)[0]
        if self._num_microbatches is None:
            self._num_microbatches = batch_size
        microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])
        if var_list is None:
            var_list = tf.compat.v1.trainable_variables(
                ) + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.
                TRAINABLE_RESOURCE_VARIABLES)

        def process_microbatch(microbatch_loss):
            """Compute clipped grads for one microbatch."""
            microbatch_loss = tf.reduce_mean(input_tensor=microbatch_loss)
            grads, _ = zip(*super(DPOptimizerClass, self).compute_gradients
                (microbatch_loss, var_list, gate_gradients,
                aggregation_method, colocate_gradients_with_ops, grad_loss))
            grads_list = [(g if g is not None else tf.zeros_like(v)) for g,
                v in zip(list(grads), var_list)]
            grads_flat = tf.nest.flatten(grads_list)
            squared_l2_norms = [tf.reduce_sum(input_tensor=tf.square(g)) for
                g in grads_flat]
            global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
            div = tf.maximum(global_norm / self._l2_norm_clip, 1.0)
            clipped_flat = [(g / div) for g in grads_flat]
            clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)
            return clipped_grads
        clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses
            )

        def reduce_noise_normalize_batch(stacked_grads):
            summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
            noise_stddev = self._l2_norm_clip * self._noise_multiplier
            noise = tf.random.normal(tf.shape(input=summed_grads), stddev=
                noise_stddev)
            noised_grads = summed_grads + noise
            return noised_grads / tf.cast(self._num_microbatches, tf.float32)
        final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
            clipped_grads)
        return list(zip(final_grads, var_list))
