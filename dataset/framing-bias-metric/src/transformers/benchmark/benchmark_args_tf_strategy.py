@property
@tf_required
def strategy(self) ->'tf.distribute.Strategy':
    return self._setup_strategy
