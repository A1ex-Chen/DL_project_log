def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    import horovod.tensorflow as hvd
    shard_mean, shard_variance = super(SyncBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)
    num_shards = hvd.size()
    if num_shards > 1:
        shard_square_of_mean = tf.math.square(shard_mean)
        shard_mean_of_square = shard_variance + shard_square_of_mean
        shard_stack = tf.stack([shard_mean, shard_mean_of_square])
        group_mean, group_mean_of_square = tf.unstack(hvd.allreduce(
            shard_stack))
        group_variance = group_mean_of_square - tf.math.square(group_mean)
        return group_mean, group_variance
    else:
        return shard_mean, shard_variance
