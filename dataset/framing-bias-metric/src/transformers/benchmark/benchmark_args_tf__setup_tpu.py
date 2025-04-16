@cached_property
@tf_required
def _setup_tpu(self) ->Tuple[
    'tf.distribute.cluster_resolver.TPUClusterResolver']:
    if self.tpu:
        try:
            if self.tpu_name:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self
                    .tpu_name)
            else:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            tpu = None
    return tpu
