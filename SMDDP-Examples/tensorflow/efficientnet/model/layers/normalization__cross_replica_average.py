def _cross_replica_average(self, t: tf.Tensor, num_shards_per_group: int):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
        if num_shards % num_shards_per_group != 0:
            raise ValueError(
                'num_shards: %d mod shards_per_group: %d, should be 0' % (
                num_shards, num_shards_per_group))
        num_groups = num_shards // num_shards_per_group
        group_assignment = [[x for x in range(num_shards) if x //
            num_shards_per_group == y] for y in range(num_groups)]
    return tf1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)
