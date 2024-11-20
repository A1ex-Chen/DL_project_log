def abstention_acc_class_i_metric(nb_classes, class_i):
    """Function to estimate accuracy over the class i prediction after removing the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    class_i : int
        Index of the class to estimate accuracy after removing abstention samples
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
        ytrueint = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i), 'int64')
        mask_pred = K.cast(K.not_equal(K.argmax(y_pred, axis=-1),
            nb_classes), 'int64')
        total_true_i = K.sum(ytrueint * mask_pred)
        true_i_pred = K.sum(mask_pred * ytrueint * K.cast(K.equal(K.argmax(
            y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))
        acc = true_i_pred / total_true_i
        condition = K.greater(total_true_i, 0)
        return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))
    metric.__name__ = 'abstention_acc_class_{}'.format(class_i)
    return metric
