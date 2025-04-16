def _value_function(self, x_input, y_true, y_pred):
    """Return classification accuracy of input"""
    return F.multitask_accuracy(y_true, y_pred)
