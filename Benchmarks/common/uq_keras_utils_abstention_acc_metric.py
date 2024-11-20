def abstention_acc_metric(nb_classes):
    """Abstained accuracy:
        Function to estimate accuracy over the predicted samples
        after removing the samples where the model is abstaining.

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
        true_pred = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.
            argmax(y_pred, axis=-1)), 'int64'))
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1),
            nb_classes), 'int64'))
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.
            argmax(y_pred, axis=-1)), 'int64'))
        condition = K.greater(total_pred, total_abs)
        abs_acc = K.switch(condition, true_pred / (total_pred - total_abs),
            total_pred / total_pred)
        return abs_acc
    metric.__name__ = 'abstention_acc'
    return metric
