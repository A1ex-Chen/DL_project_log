def sampling(self, x=64):
    z = tf.random.normal(shape=(x, self.n_latent_features))
    z = self.fc3(z)
    return self.decoder(z)
