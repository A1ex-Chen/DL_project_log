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
