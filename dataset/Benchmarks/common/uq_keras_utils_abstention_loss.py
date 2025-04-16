def abstention_loss(alpha, mask):
    """Function to compute abstention loss.
        It is composed by two terms:
        (i) original loss of the multiclass classification problem,
        (ii) cost associated to the abstaining samples.

    Parameters
    ----------
    alpha : Keras variable
        Weight of abstention term in cost function
    mask : ndarray
        Numpy array to use as mask for abstention:
        it is 1 on the output associated to the abstention class and 0 otherwise
    """

    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        base_pred = (1 - mask) * y_pred + K.epsilon()
        base_true = y_true
        base_cost = K.categorical_crossentropy(base_true, base_pred)
        abs_pred = K.sum(mask * y_pred, axis=-1)
        abs_pred = K.clip(abs_pred, K.epsilon(), 1.0 - K.epsilon())
        return (1.0 - abs_pred) * base_cost - alpha * K.log(1.0 - abs_pred)
    loss.__name__ = 'abs_crossentropy'
    return loss
