def _value_function(self, x_input, y_true, y_pred, reduce=None):
    """Return loss value of input"""
    return F.multitask_loss(y_true, y_pred, criterion=self.criterion,
        reduce=reduce)
