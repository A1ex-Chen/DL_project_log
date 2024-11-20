def xent(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)
