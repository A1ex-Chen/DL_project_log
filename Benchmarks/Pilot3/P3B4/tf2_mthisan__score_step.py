@tf.function
def _score_step(self, text, labels):
    predictions = self.model(text, training=False)
    loss = 0
    for i in range(self.num_tasks):
        loss += self.loss_object(labels[i], predictions[i]) / self.num_tasks
    return predictions, loss
