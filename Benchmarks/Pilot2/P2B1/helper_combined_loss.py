def combined_loss(y_true, y_pred):
    """
    Uses a combination of mean_squared_error and an L1 penalty on the output of AE
    """
    return mse(y_true, y_pred) + 0.01 * mae(0, y_pred)
