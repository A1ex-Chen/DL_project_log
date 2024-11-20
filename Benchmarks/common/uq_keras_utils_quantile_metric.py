def quantile_metric(nout, index, quantile):
    """This function computes the quantile metric for a given quantile and corresponding output index. This is provided as a metric to track evolution while training.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    index : int
        Index of output corresponding to the given quantile.
    quantile: float in (0, 1)
        Fraction corresponding to the quantile
    """

    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a quantile model. The predictions follow the order: (q50_0, qlow_0, qhigh_0, q50_1, qlow_1, qhigh_1, ...) with q50_i the median of the ith output and qlow_i and qhigh_i the low and high specified quantiles of the ith output.
        """
        y_shape = K.shape(y_true)
        if nout > 1:
            y_qtl = K.reshape(y_pred[:, index::3], y_shape)
        else:
            y_qtl = K.reshape(y_pred[:, index], y_shape)
        return quantile_loss(quantile, y_true, y_qtl)
    metric.__name__ = 'quantile_{}'.format(quantile)
    return metric
