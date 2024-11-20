def get_metrics(one_hot: bool):
    """Get a dict of available metrics to track."""
    if one_hot:
        return {'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'
            ), 'accuracy': tf.keras.metrics.CategoricalAccuracy(name=
            'accuracy'), 'top_1': tf.keras.metrics.CategoricalAccuracy(name
            ='accuracy'), 'top_5': tf.keras.metrics.TopKCategoricalAccuracy
            (k=5, name='top_5_accuracy')}
    else:
        return {'acc': tf.keras.metrics.SparseCategoricalAccuracy(name=
            'accuracy'), 'accuracy': tf.keras.metrics.
            SparseCategoricalAccuracy(name='accuracy'), 'top_1': tf.keras.
            metrics.SparseCategoricalAccuracy(name='accuracy'), 'top_5': tf
            .keras.metrics.SparseTopKCategoricalAccuracy(k=5, name=
            'top_5_accuracy')}
