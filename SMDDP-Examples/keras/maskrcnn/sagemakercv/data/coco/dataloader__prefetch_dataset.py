def _prefetch_dataset(filename):
    return tf.data.TFRecordDataset(filename).prefetch(1)
