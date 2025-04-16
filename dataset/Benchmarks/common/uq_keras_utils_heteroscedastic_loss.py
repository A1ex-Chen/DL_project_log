def heteroscedastic_loss(nout):
    """This function computes the heteroscedastic loss for the heteroscedastic model. Both mean and standard deviation predictions are taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """

    def loss(y_true, y_pred):
        """This function computes the heteroscedastic loss.

        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        y_shape = K.shape(y_true)
        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], y_shape)
            log_sig2 = K.reshape(y_pred[:, 1::nout], y_shape)
        else:
            y_out = K.reshape(y_pred[:, 0], y_shape)
            log_sig2 = K.reshape(y_pred[:, 1], y_shape)
        diff_sq = K.square(y_out - y_true)
        return K.mean(K.exp(-log_sig2) * diff_sq + log_sig2)
    return loss
