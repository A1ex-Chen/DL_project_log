def get_train_tfdataset(self) ->tf.data.Dataset:
    """
        Returns the training :class:`~tf.data.Dataset`.

        Subclass and override this method if you want to inject some custom behavior.
        """
    if self.train_dataset is None:
        raise ValueError('Trainer: training requires a train_dataset.')
    self.total_train_batch_size = (self.args.train_batch_size * self.args.
        gradient_accumulation_steps)
    self.num_train_examples = tf.data.experimental.cardinality(self.
        train_dataset).numpy()
    if self.num_train_examples < 0:
        raise ValueError(
            'The training dataset must have an asserted cardinality')
    ds = self.train_dataset.repeat().shuffle(self.num_train_examples, seed=
        self.args.seed).batch(self.total_train_batch_size, drop_remainder=
        self.args.dataloader_drop_last).prefetch(tf.data.experimental.AUTOTUNE)
    return self.args.strategy.experimental_distribute_dataset(ds)
