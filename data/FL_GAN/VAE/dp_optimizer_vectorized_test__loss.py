def _loss(self, val0, val1):
    """Loss function that is minimized at the mean of the input points."""
    return 0.5 * tf.reduce_sum(input_tensor=tf.math.squared_difference(val0,
        val1), axis=1)
