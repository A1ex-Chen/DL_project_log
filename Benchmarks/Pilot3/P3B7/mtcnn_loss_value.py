def loss_value(self, y_pred, y_true, reduce='sum'):
    """Compute the cross entropy loss"""
    losses = {}
    for key, value in y_true.items():
        losses[key] = F.cross_entropy(y_pred[key], y_true[key])
    if reduce:
        total = 0
        for _, loss in losses.items():
            total += loss
        if reduce == 'mean':
            losses = total / len(losses)
        elif reduce == 'sum':
            losses = total
    return losses
