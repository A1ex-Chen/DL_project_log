@cached_property
@tf_required
def _setup_strategy(self) ->Tuple['tf.distribute.Strategy',
    'tf.distribute.cluster_resolver.TPUClusterResolver']:
    if self.is_tpu:
        tf.config.experimental_connect_to_cluster(self._setup_tpu)
        tf.tpu.experimental.initialize_tpu_system(self._setup_tpu)
        strategy = tf.distribute.experimental.TPUStrategy(self._setup_tpu)
    elif self.is_gpu:
        tf.config.experimental.set_visible_devices(self.gpu_list[self.
            device_idx], 'GPU')
        strategy = tf.distribute.OneDeviceStrategy(device=
            f'/gpu:{self.device_idx}')
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
        strategy = tf.distribute.OneDeviceStrategy(device=
            f'/cpu:{self.device_idx}')
    return strategy
