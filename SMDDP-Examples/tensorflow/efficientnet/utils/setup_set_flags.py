def set_flags(params):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    if params.intraop_threads:
        os.environ['TF_NUM_INTRAOP_THREADS'] = params.intraop_threads
    if params.interop_threads:
        os.environ['TF_NUM_INTEROP_THREADS'] = params.interop_threads
    if params.use_xla:
        os.environ['TF_XLA_FLAGS'
            ] = '--tf_xla_enable_lazy_compilation=false --tf_xla_auto_jit=1'
        os.environ['TF_EXTRA_PTXAS_OPTIONS'] = '-sw200428197=true'
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        assert tf.config.experimental.get_memory_growth(gpu)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()],
            'GPU')
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    if params.use_amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16',
            loss_scale='dynamic')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
