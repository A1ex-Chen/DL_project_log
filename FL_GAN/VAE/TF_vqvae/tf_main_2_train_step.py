def train_step(self, x):
    with tf.GradientTape() as tape:
        reconstructions = self.vqvae(x)
        reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2
            ) / self.train_variance
        total_loss = reconstruction_loss + sum(self.vqvae.losses)
    grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
    with tf.GradientTape() as tape2:
        extra_time = time.time()
        encoder = self.vqvae.get_layer('encoder')
        quantizer = self.vqvae.get_layer('vector_quantizer')
        encoded_outputs = encoder(x).numpy()
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.
            shape[-1])
        codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
        codebook_indices = codebook_indices.numpy().reshape(encoded_outputs
            .shape[:-1])
        extra_time = time.time() - extra_time
        ar = self.pixel_cnn(codebook_indices)
        ar_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            codebook_indices, ar)
    ar_gradients = tape2.gradient(ar_loss, self.pixel_cnn.trainable_variables)
    self.optimizer.apply_gradients(zip(ar_gradients, self.pixel_cnn.
        trainable_variables))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
    self.total_pixel_cnn_loss_tracker.update_state(ar_loss)
    self.acc_tracker.update_state(y_true=codebook_indices, y_pred=tf.argmax
        (ar, 3))
    self.extra_time.update_state(extra_time)
    return {'loss': self.total_loss_tracker.result(), 'reconstruction_loss':
        self.reconstruction_loss_tracker.result(), 'vqvae_loss': self.
        vq_loss_tracker.result(), 'Pixel loss': self.
        total_pixel_cnn_loss_tracker.result(), 'Pixel accuracy': self.
        acc_tracker.result(), 'Extra time': self.extra_time.result()}
