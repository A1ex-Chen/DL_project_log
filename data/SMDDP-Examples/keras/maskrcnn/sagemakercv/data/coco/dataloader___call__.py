def __call__(self, input_context=None):
    batch_size = self._batch_size
    do_dist_eval = self._dist_eval
    try:
        seed = self._seed if not MPI_is_distributed(
            ) else self._seed * MPI_rank()
    except (KeyError, TypeError):
        seed = None
    if MPI_is_distributed():
        n_gpus = MPI_size()
    elif input_context is not None:
        n_gpus = input_context.num_input_pipelines
    else:
        n_gpus = 1
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
    if self._mode == tf.estimator.ModeKeys.TRAIN:
        if input_context is not None:
            logging.info('Using Dataset Sharding with TF Distributed')
            _num_shards = input_context.num_input_pipelines
            _shard_idx = input_context.input_pipeline_id
        elif MPI_is_distributed():
            logging.info('Using Dataset Sharding')
            _shard_idx, _num_shards = MPI_rank_and_size()
        try:
            dataset = dataset.shard(num_shards=_num_shards, index=_shard_idx)
            dataset = dataset.shuffle(math.ceil(512 / _num_shards))
        except NameError:
            pass
    elif do_dist_eval and (self._mode == tf.estimator.ModeKeys.PREDICT or 
        self._mode == tf.estimator.ModeKeys.EVAL):
        if MPI_is_distributed():
            logging.info('Using Evaluation Dataset Sharding')
            _shard_idx, _num_shards = MPI_rank_and_size()
            max_shards = min(_num_shards, 512)
            try:
                dataset = dataset.shard(num_shards=max_shards, index=
                    _shard_idx % max_shards)
            except NameError:
                pass

    def _prefetch_dataset(filename):
        return tf.data.TFRecordDataset(filename).prefetch(1)
    dataset = dataset.interleave(map_func=_prefetch_dataset, cycle_length=
        64, block_length=8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._num_examples is not None and self._num_examples > 0:
        logging.info('[*] Limiting the amount of sample to: %d' % self.
            _num_examples)
        dataset = dataset.take(self._num_examples)
    dataset = dataset.cache()
    if self._mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=4096,
            reshuffle_each_iteration=True, seed=seed)
        dataset = dataset.repeat()
    dataset = dataset.map(map_func=self._create_dataset_parser_fn(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    if self._use_fake_data:
        logging.info('Using Fake Dataset Loop...')
        dataset = dataset.take(1).cache().repeat()
        if self._mode != tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.take(int(5000 / batch_size))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    """if self._mode == tf.estimator.ModeKeys.PREDICT or n_gpus > 1:
            if not tf.distribute.has_strategy():
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device(
                        '/gpu:0',  # With Horovod the local GPU is always 0
                        buffer_size=1,
                    )
                )"""
    if not self._disable_options:
        data_options = tf.data.Options()
        data_options.experimental_deterministic = seed is not None
        if LooseVersion(tf.__version__) <= LooseVersion('2.0.0'):
            data_options.experimental_distribute.auto_shard = False
        else:
            data_options.experimental_distribute.auto_shard_policy = (tf.
                data.experimental.AutoShardPolicy.OFF)
        data_options.experimental_slack = self._data_slack
        data_options.experimental_threading.max_intra_op_parallelism = 1
        (data_options.experimental_optimization.apply_default_optimizations
            ) = False
        data_options.experimental_optimization.filter_fusion = True
        data_options.experimental_optimization.map_and_batch_fusion = True
        data_options.experimental_optimization.map_and_filter_fusion = True
        data_options.experimental_optimization.map_fusion = True
        data_options.experimental_optimization.map_parallelization = True
        if int(tf.__version__.split('.')[1]) < 6:
            map_vectorization_options = (tf.data.experimental.
                MapVectorizationOptions())
            map_vectorization_options.enabled = True
            map_vectorization_options.use_choose_fastest = True
            data_options.experimental_optimization.map_vectorization = (
                map_vectorization_options)
        data_options.experimental_optimization.noop_elimination = True
        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_optimization.shuffle_and_repeat_fusion = True
        dataset = dataset.with_options(data_options)
    return dataset
