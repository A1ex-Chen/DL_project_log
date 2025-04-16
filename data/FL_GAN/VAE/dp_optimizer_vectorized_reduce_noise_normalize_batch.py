def reduce_noise_normalize_batch(stacked_grads):
    summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
    noise_stddev = self._l2_norm_clip * self._noise_multiplier
    noise = tf.random.normal(tf.shape(input=summed_grads), stddev=noise_stddev)
    noised_grads = summed_grads + noise
    return noised_grads / tf.cast(self._num_microbatches, tf.float32)
