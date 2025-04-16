def __init__(self, train_variance, dataset, data_shape=[], **kwargs):
    super().__init__(**kwargs)
    self.train_variance = train_variance
    self.data_shape = data_shape
    self.vqvae = VAE(dataset)
    self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker = keras.metrics.Mean(name=
        'reconstruction_loss')
    self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
