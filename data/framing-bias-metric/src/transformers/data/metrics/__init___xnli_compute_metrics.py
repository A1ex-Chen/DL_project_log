def xnli_compute_metrics(task_name, preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_sklearn(xnli_compute_metrics)
    assert len(preds) == len(labels
        ), f'Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}'
    if task_name == 'xnli':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
