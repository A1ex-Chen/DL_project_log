def accuracy(preds, labels):
    return (preds == labels).mean()
