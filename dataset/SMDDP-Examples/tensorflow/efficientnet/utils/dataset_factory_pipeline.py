def pipeline(self, dataset: tf.data.Dataset) ->tf.data.Dataset:
    """Build a pipeline fetching, shuffling, and preprocessing the dataset.

    Args:
      dataset: A `tf.data.Dataset` that loads raw files.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if self._num_gpus > 1:
        dataset = dataset.shard(self._num_gpus, sdp.rank())
    if self.is_training:
        dataset.shuffle(buffer_size=self._file_shuffle_buffer_size)
    if self.is_training and not self._cache:
        dataset = dataset.repeat()
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10,
        block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._cache:
        dataset = dataset.cache()
    if self.is_training:
        dataset = dataset.shuffle(self._shuffle_buffer_size)
        dataset = dataset.repeat()
    preprocess = self.parse_record
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.
        experimental.AUTOTUNE)
    if self._num_gpus > 1:
        dataset = dataset.batch(self.local_batch_size, drop_remainder=self.
            is_training)
    else:
        dataset = dataset.batch(self.global_batch_size, drop_remainder=self
            .is_training)
    mixup_alpha = self.mixup_alpha if self.is_training else 0.0
    dataset = dataset.map(functools.partial(self.mixup, self.
        local_batch_size, mixup_alpha), num_parallel_calls=64)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
