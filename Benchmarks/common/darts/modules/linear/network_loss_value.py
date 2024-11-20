def loss_value(self, x_data, y_true, y_pred, reduce='mean'):
    """Calculate a value of loss function"""
    y_pred = self(x_data)
    losses = {}
    for key, value in y_true.items():
        losses[key] = F.nll_loss(F.log_softmax(y_pred[key], dim=1), y_true[key]
            )
    if reduce:
        total = 0
        for _, value in losses.items():
            total += value
        if reduce == 'mean':
            losses = total / len(losses)
        elif reduce == 'sum':
            losses = total
    return losses
