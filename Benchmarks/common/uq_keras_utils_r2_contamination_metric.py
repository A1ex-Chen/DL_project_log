def r2_contamination_metric(nout):
    """This function computes the r2 for the contamination model. The r2 is computed over the prediction. Therefore, the augmentation for the index variable is ignored.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    """

    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth. Since the keras tensor includes an extra column to store the index of the data sample in the training set this column is ignored.
        y_pred : Keras tensor
            Keras tensor with the predictions of the contamination model (no data index).
        """
        y_true_ = K.reshape(y_true[:, :-1], K.shape(y_pred))
        SS_res = K.sum(K.square(y_true_ - y_pred))
        SS_tot = K.sum(K.square(y_true_ - K.mean(y_true_)))
        return 1.0 - SS_res / (SS_tot + K.epsilon())
    metric.__name__ = 'r2_contamination'
    return metric
