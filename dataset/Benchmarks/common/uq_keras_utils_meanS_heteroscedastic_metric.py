def meanS_heteroscedastic_metric(nout):
    """This function computes the mean log of the variance (log S) for the heteroscedastic model. The mean log is computed over the standard deviation prediction and the mean prediction is not taken into account.

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
            log_sig2 = y_pred[:, 1::nout]
        else:
            log_sig2 = y_pred[:, 1]
        return K.mean(log_sig2)
    metric.__name__ = 'meanS_heteroscedastic'
    return metric
