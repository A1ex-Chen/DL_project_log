@staticmethod
def _compute_nb_instances(batch):
    labels = batch[-1]
    if isinstance(labels, PerReplica):
        labels = tf.concat(labels.values, axis=0)
    nb_instances = tf.reduce_sum(tf.cast(labels != -100, dtype=tf.int32))
    return nb_instances
