def default_compute_objective(metrics: Dict[str, float]) ->float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop('eval_loss', None)
    _ = metrics.pop('epoch', None)
    return loss if len(metrics) == 0 else sum(metrics.values())
