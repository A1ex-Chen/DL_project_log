def get_data(args):
    print('Preparing data...')
    data, dataset_info = tfds.load(name='imagenet2012', with_info=True,
        as_supervised=True, download=False, data_dir=args.data_dir)
    train_batches = data['train'].shuffle(buffer_size=10000).map(
        preprocess_training_image, num_parallel_calls=tf.data.experimental.
        AUTOTUNE).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE
        )
    validation_batches = data['validation'].map(preprocess_validation_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(args.
        batch_size * 3).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches, dataset_info
