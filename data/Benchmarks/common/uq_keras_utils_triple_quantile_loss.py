def triple_quantile_loss(nout, lowquantile, highquantile):
    """This function computes the quantile loss for the median and low and high quantiles. The median is given twice the weight of the other components.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    lowquantile: float in (0, 1)
        Fraction corresponding to the low quantile
    highquantile: float in (0, 1)
        Fraction corresponding to the high quantile
    """

    def loss(y_true, y_pred):
        """This function computes the quantile loss, considering the median and low and high quantiles.

        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (q50_0, qlow_0, qhigh_0, q50_1, qlow_1, qhigh_1, ...) with q50_i the median of the ith output and qlow_i and qhigh_i the low and high specified quantiles of the ith output.
        """
        y_shape = K.shape(y_true)
        if nout > 1:
            y_qtl0 = K.reshape(y_pred[:, 0::3], y_shape)
            y_qtl1 = K.reshape(y_pred[:, 1::3], y_shape)
            y_qtl2 = K.reshape(y_pred[:, 2::3], y_shape)
        else:
            y_qtl0 = K.reshape(y_pred[:, 0], y_shape)
            y_qtl1 = K.reshape(y_pred[:, 1], y_shape)
            y_qtl2 = K.reshape(y_pred[:, 2], y_shape)
        return quantile_loss(lowquantile, y_true, y_qtl1) + quantile_loss(
            highquantile, y_true, y_qtl2) + 2.0 * quantile_loss(0.5, y_true,
            y_qtl0)
    return loss
