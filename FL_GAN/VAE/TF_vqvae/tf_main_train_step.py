def train_step(self, x):
    with tf.GradientTape() as tape:
        reconstructions = self.vqvae(x)
        reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2
            ) / self.train_variance
        total_loss = reconstruction_loss + sum(self.vqvae.losses)
    grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
    return {'loss': self.total_loss_tracker.result(), 'reconstruction_loss':
        self.reconstruction_loss_tracker.result(), 'vqvae_loss': self.
        vq_loss_tracker.result()}
