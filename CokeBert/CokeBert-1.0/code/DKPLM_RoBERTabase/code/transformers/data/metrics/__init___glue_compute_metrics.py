def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
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
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'mnli-mm':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'qnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'rte':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'wnli':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
