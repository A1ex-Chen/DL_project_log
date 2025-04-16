def _log_confusion_matrix(validator):
    """Logs the confusion matrix for the given validator using DVCLive."""
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == 'detect':
        names += ['background']
    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)
    live.log_sklearn_plot('confusion_matrix', targets, preds, name=
        'cf.json', normalized=True)
