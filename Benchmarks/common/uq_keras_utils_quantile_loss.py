def quantile_loss(quantile, y_true, y_pred):
    """This function computes the quantile loss for a given quantile fraction.

    Parameters
    ----------
    quantile : float in (0, 1)
        Quantile fraction to compute the loss.
    y_true : Keras tensor
        Keras tensor including the ground truth
    y_pred : Keras tensor
        Keras tensor including the predictions of a quantile model.
    """
    error = y_true - y_pred
    return K.mean(K.maximum(quantile * error, (quantile - 1) * error))
