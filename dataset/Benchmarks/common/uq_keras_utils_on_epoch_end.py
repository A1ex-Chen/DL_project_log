def on_epoch_end(self, epoch, logs={}):
    """Updates the parameters of the distributions in the contamination model on epoch end. The parameters updated are: 'a' for the global weight of the membership to the normal distribution, 'sigmaSQ' for the variance of the normal distribution and 'gammaSQ' for the scale of the Cauchy distribution of outliers. The latent variables are updated as well: 'T_k' describing in the first column the probability of membership to normal distribution and in the second column probability of membership to the Cauchy distribution i.e. outlier. Stores evolution of global parameters (a, sigmaSQ and gammaSQ).

        Parameters
        ----------
        epoch : integer
            Current epoch in training.
        logs : keras logs
            Metrics stored during current keras training.
        """
    y_pred = self.model.predict(self.x)
    error = self.y.squeeze() - y_pred.squeeze()
    errorSQ = error ** 2
    aux = np.mean(self.T[:, 0])
    if aux > self.a_max:
        aux = self.a_max
    K.set_value(self.a, aux)
    K.set_value(self.sigmaSQ, np.sum(self.T[:, 0] * errorSQ) / np.sum(self.
        T[:, 0]))
    gmSQ_eval = K.get_value(self.gammaSQ)
    grad_gmSQ = (0.5 * np.sum(self.T[:, 1]) - np.sum(self.T[:, 1] * errorSQ /
        (gmSQ_eval + errorSQ))) / gmSQ_eval
    eta = K.get_value(self.model.optimizer.lr)
    new_gmSQ = gmSQ_eval - eta * grad_gmSQ
    while new_gmSQ < 0 or new_gmSQ / gmSQ_eval > 1000:
        eta /= 2
        new_gmSQ = gmSQ_eval - eta * grad_gmSQ
    K.set_value(self.gammaSQ, new_gmSQ)
    a_eval = K.get_value(self.a)
    sigmaSQ_eval = K.get_value(self.sigmaSQ)
    gammaSQ_eval = K.get_value(self.gammaSQ)
    print('a: %f, sigmaSQ: %f, gammaSQ: %f' % (a_eval, sigmaSQ_eval,
        gammaSQ_eval))
    norm_eval = norm.pdf(error, loc=0, scale=np.sqrt(sigmaSQ_eval))
    cauchy_eval = cauchy.pdf(error, loc=0, scale=np.sqrt(gammaSQ_eval))
    denominator = a_eval * norm_eval + (1.0 - a_eval) * cauchy_eval
    self.T[:, 0] = a_eval * norm_eval / denominator
    self.T[:, 1] = (1.0 - a_eval) * cauchy_eval / denominator
    K.set_value(self.T_k, self.T)
    self.avalues.append(a_eval)
    self.sigmaSQvalues.append(sigmaSQ_eval)
    self.gammaSQvalues.append(gammaSQ_eval)
