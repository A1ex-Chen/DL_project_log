def simple_accuracy(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_sklearn(simple_accuracy)
    return (preds == labels).mean()
