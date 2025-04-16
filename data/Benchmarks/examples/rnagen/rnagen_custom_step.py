def custom_step(self, data, train=False):
    with tf.GradientTape() as tape:
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        if type(x) == tuple:
            y_pred = self.decoder([z, x[1]])
        else:
            y_pred = self.decoder(z)
        reconstruction_loss = keras.losses.binary_crossentropy(y, y_pred)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
    if train:
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.compiled_metrics.update_state(y, y_pred)
    self.total_loss_tracker.update_state(total_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    results = {'loss': self.total_loss_tracker.result(), 'kl': self.
        kl_loss_tracker.result()}
    for m in self.metrics:
        results[m.name] = m.result()
    return results
