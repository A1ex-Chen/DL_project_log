def mse_contamination_metric(nout):
    """This function computes the mean squared error (mse) for the contamination model. The mse is computed over the prediction. Therefore, the augmentation for the index variable is ignored.

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
        return mean_squared_error(y_true[:, :nout], y_pred[:, :nout])
    metric.__name__ = 'mse_contamination'
    return metric
