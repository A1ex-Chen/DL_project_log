def contamination_loss(nout, T_k, a, sigmaSQ, gammaSQ):
    """Function to compute contamination loss. It is composed by two terms: (i) the loss with respect to the normal distribution that models the distribution of the training data samples, (ii) the loss with respect to the Cauchy distribution that models the distribution of the outlier samples. Note that the evaluation of this contamination loss function does not make sense for any data different to the training set. This is because latent variables are only defined for samples in the training set.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    T_k : Keras tensor
        Tensor containing latent variables (probability of membership to normal and Cauchy distributions) for each of the samples in the training set. (Validation data is usually augmented too to be able to run training with validation set, however loss in validation should not be used as a criterion for early stopping training since the latent variables are defined for the training only, and thus, are not valid when used in combination with data different from training).
    a : Keras variable
        Probability of belonging to the normal distribution
    sigmaSQ : Keras variable
        Variance estimated for the normal distribution
    gammaSQ : Keras variable
        Scale estimated for the Cauchy distribution
    """

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
        term_cauchy = K.log(1.0 + diff_sq / gammaSQ) + 0.5 * K.log(piSQ *
            gammaSQ) - K.log(1.0 - a)
        batch_index = K.cast(y_true[:, -1], 'int64')
        T_0_red = K.gather(T_k[:, 0], batch_index)
        T_1_red = K.gather(T_k[:, 1], batch_index)
        return K.sum(T_0_red * term_normal + T_1_red * term_cauchy)
    return loss
