@property
@tf_required
def strategy(self) ->'tf.distribute.Strategy':
    """
        The strategy used for distributed training.
        """
    return self._setup_strategy
