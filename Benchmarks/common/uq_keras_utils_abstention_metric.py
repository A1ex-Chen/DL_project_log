def abstention_metric(nb_classes):
    """Function to estimate fraction of the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """

    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1),
            nb_classes), 'int64'))
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.
            argmax(y_pred, axis=-1)), 'int64'))
        return total_abs / total_pred
    metric.__name__ = 'abstention'
    return metric
