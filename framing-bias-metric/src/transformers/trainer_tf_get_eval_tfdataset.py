def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset]=None
    ) ->tf.data.Dataset:
    """
        Returns the evaluation :class:`~tf.data.Dataset`.

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                If provided, will override `self.eval_dataset`. The dataset should yield tuples of ``(features,
                labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is
                a tensor, the loss is calculated by the model by calling ``model(features, labels=labels)``. If
                ``labels`` is a dict, such as when using a QuestionAnswering head model with multiple targets, the loss
                is instead calculated by calling ``model(features, **labels)``.

        Subclass and override this method if you want to inject some custom behavior.
        """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError('Trainer: evaluation requires an eval_dataset.')
    eval_dataset = (eval_dataset if eval_dataset is not None else self.
        eval_dataset)
    num_examples = tf.data.experimental.cardinality(eval_dataset).numpy()
    if num_examples < 0:
        raise ValueError(
            'The training dataset must have an asserted cardinality')
    approx = math.floor if self.args.dataloader_drop_last else math.ceil
    steps = approx(num_examples / self.args.eval_batch_size)
    ds = eval_dataset.repeat().batch(self.args.eval_batch_size,
        drop_remainder=self.args.dataloader_drop_last).prefetch(tf.data.
        experimental.AUTOTUNE)
    return self.args.strategy.experimental_distribute_dataset(ds
        ), steps, num_examples
