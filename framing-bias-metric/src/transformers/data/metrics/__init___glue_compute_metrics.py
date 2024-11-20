def glue_compute_metrics(task_name, preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_sklearn(glue_compute_metrics)
    assert len(preds) == len(labels
        ), f'Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}'
    if task_name == 'cola':
        return {'mcc': matthews_corrcoef(labels, preds)}
    elif task_name == 'sst-2':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'mrpc':
        return acc_and_f1(preds, labels)
    elif task_name == 'sts-b':
        return pearson_and_spearman(preds, labels)
    elif task_name == 'qqp':
        return acc_and_f1(preds, labels)
    elif task_name == 'mnli':
        return {'mnli/acc': simple_accuracy(preds, labels)}
    elif task_name == 'mnli-mm':
        return {'mnli-mm/acc': simple_accuracy(preds, labels)}
    elif task_name == 'qnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'rte':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'wnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'hans':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
