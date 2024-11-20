def mae_heteroscedastic_metric(nout):
    """This function computes the mean absolute error (mae) for the heteroscedastic model. The mae is computed over the prediction of the mean and the standard deviation prediction is not taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """

    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], K.shape(y_true))
        else:
            y_out = K.reshape(y_pred[:, 0], K.shape(y_true))
        return mean_absolute_error(y_true, y_out)
    metric.__name__ = 'mae_heteroscedastic'
    return metric
