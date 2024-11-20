def swap_weights(self):
    """Swap the average and moving weights.

    This is a convenience method to allow one to evaluate the averaged weights
    at test time. Loads the weights stored in `self._average` into the model,
    keeping a copy of the original model weights. Swapping twice will return
    the original weights.
    """
    if tf.distribute.in_cross_replica_context():
        strategy = tf.distribute.get_strategy()
        strategy.run(self._swap_weights, args=())
    else:
        raise ValueError(
            'Swapping weights must occur under a tf.distribute.Strategy')
