def __init__(self, train_variance, num_embeddings=128, embedding_dim=256,
    data_shape=[], **kwargs):
    super().__init__(**kwargs)
    self.train_variance = train_variance
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.data_shape = data_shape
    self.vqvae = get_vqvae(self.embedding_dim, self.num_embeddings, data_shape)
    self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker = keras.metrics.Mean(name=
        'reconstruction_loss')
    self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')
