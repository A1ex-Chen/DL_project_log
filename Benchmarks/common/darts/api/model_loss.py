def loss(self, x_data, y_true, reduce='mean'):
    """Forward propagate network and return a value of loss function"""
    if reduce not in (None, 'sum', 'mean'):
        raise ValueError('`reduce` must be either None, `sum`, or `mean`!')
    y_pred = self(x_data)
    return y_pred, self.loss_value(x_data, y_true, y_pred, reduce=reduce)
