def loss(y_true, y_pred):
    """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict. It is assumed that this keras tensor includes extra columns to store the index of the data sample in the training set.
        y_pred : keras tensor
            Prediction made by the model.
        """
    y_shape = K.shape(y_pred)
    y_true_ = K.reshape(y_true[:, :-1], y_shape)
    if nout > 1:
        diff_sq = K.sum(K.square(y_true_ - y_pred), axis=-1)
    else:
        diff_sq = K.square(y_true_ - y_pred)
    term_normal = diff_sq / (2.0 * sigmaSQ) + 0.5 * K.log(sigmaSQ
        ) + 0.5 * K.log(2.0 * np.pi) - K.log(a)
    term_cauchy = K.log(1.0 + diff_sq / gammaSQ) + 0.5 * K.log(piSQ * gammaSQ
        ) - K.log(1.0 - a)
    batch_index = K.cast(y_true[:, -1], 'int64')
    T_0_red = K.gather(T_k[:, 0], batch_index)
    T_1_red = K.gather(T_k[:, 1], batch_index)
    return K.sum(T_0_red * term_normal + T_1_red * term_cauchy)
