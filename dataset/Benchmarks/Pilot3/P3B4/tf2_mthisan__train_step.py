@tf.function
def _train_step(self, text, labels):
    with tf.GradientTape() as tape:
        predictions = self.model(text, training=True)
        loss = 0
        for i in range(self.num_tasks):
            loss += self.loss_object(labels[i], predictions[i]
                ) / self.num_tasks
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.
        trainable_variables))
    return predictions, loss
