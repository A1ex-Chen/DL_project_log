def sampling(params):
    z_mean_, z_log_var_ = params
    batch_size = K.shape(z_mean_)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.0,
        stddev=epsilon_std)
    return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
