def __init__(self, train_variance, embedding_dim=64, num_embeddings=128,
    data_shape=[], **kwargs):
    super().__init__(**kwargs)
    self.train_variance = train_variance
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.data_shape = data_shape
    self.vqvae = get_vqvae(self.embedding_dim, self.num_embeddings, data_shape)
    self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker = keras.metrics.Mean(name=
        'reconstruction_loss')
    self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
    self.total_pixel_cnn_loss_tracker = keras.metrics.Mean(name=
        'total_cnn_loss')
    self.acc_tracker = keras.metrics.Accuracy(name='acc')
    self.extra_time = keras.metrics.Sum(name='extra time')
    self.pixel_cnn = get_pixel_cnn([int(data_shape[0] / 4), int(data_shape[
        0] / 4)], num_embeddings)
