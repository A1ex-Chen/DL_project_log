def abstention_class_i_metric(nb_classes, class_i):
    """Function to estimate fraction of the samples where the model is abstaining in class i.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    class_i : int
        Index of the class to estimate accuracy
    """

    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        ytrue_i_int = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i),
            'int64')
        total_class_i = K.sum(ytrue_i_int)
        y_abs = K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64')
        total_abs_i = K.sum(ytrue_i_int * y_abs)
        abs_i = total_abs_i / total_class_i
        condition = K.greater(total_class_i, 0)
        return K.switch(condition, abs_i, K.zeros_like(abs_i, dtype=abs_i.
            dtype))
    metric.__name__ = 'abstention_class_{}'.format(class_i)
    return metric
