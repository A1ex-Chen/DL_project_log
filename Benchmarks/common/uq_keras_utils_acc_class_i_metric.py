def acc_class_i_metric(class_i):
    """Function to estimate accuracy over the ith class prediction.
        This estimation is global (i.e. abstaining samples are not removed)

    Parameters
    ----------
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
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        ytrueint = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i), 'int64')
        total_true_i = K.sum(ytrueint)
        ypredint = K.cast(K.equal(K.argmax(y_pred[:, :-1], axis=-1),
            class_i), 'int64')
        true_i_pred = K.sum(ytrueint * ypredint)
        acc = true_i_pred / total_true_i
        condition = K.greater(total_true_i, 0)
        return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))
    metric.__name__ = 'acc_class_{}'.format(class_i)
    return metric
