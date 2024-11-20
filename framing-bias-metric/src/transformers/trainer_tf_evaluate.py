def evaluate(self, eval_dataset: Optional[tf.data.Dataset]=None) ->Dict[str,
    float]:
    """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. The dataset should yield tuples of
                ``(features, labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If
                ``labels`` is a tensor, the loss is calculated by the model by calling ``model(features,
                labels=labels)``. If ``labels`` is a dict, such as when using a QuestionAnswering head model with
                multiple targets, the loss is instead calculated by calling ``model(features, **labels)``.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
    eval_ds, steps, num_examples = self.get_eval_tfdataset(eval_dataset)
    output = self.prediction_loop(eval_ds, steps, num_examples, description
        ='Evaluation')
    logs = {**output.metrics}
    logs['epoch'] = self.epoch_logging
    self.log(logs)
    return output.metrics
