def load_model_weights(self, model_dir):
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if not latest_checkpoint:
        return 0
    self.model.load_weights(latest_checkpoint)
    return self.model.optimizer.iterations
