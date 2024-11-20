@cached_property
@tf_required
def _setup_strategy(self) ->Tuple['tf.distribute.Strategy', int]:
    logger.info('Tensorflow: setting up strategy')
    if self.xla:
        tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices('GPU')
    if self.fp16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    if self.no_cuda:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
    else:
        try:
            if self.tpu_name:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self
                    .tpu_name)
            else:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            tpu = None
        if tpu:
            if self.fp16:
                policy = tf.keras.mixed_precision.experimental.Policy(
                    'mixed_bfloat16')
                tf.keras.mixed_precision.experimental.set_policy(policy)
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        elif len(gpus) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError(
                'Cannot find the proper strategy please check your environment properties.'
                )
    return strategy
