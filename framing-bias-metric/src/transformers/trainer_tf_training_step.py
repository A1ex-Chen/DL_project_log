def training_step(self, features, labels, nb_instances_in_global_batch):
    """
        Perform a training step on features and labels.

        Subclass and override to inject some custom behavior.
        """
    per_example_loss, _ = self.run_model(features, labels, True)
    scaled_loss = per_example_loss / tf.cast(nb_instances_in_global_batch,
        dtype=per_example_loss.dtype)
    gradients = tf.gradients(scaled_loss, self.model.trainable_variables)
    gradients = [(g if g is not None else tf.zeros_like(v)) for g, v in zip
        (gradients, self.model.trainable_variables)]
    if self.args.gradient_accumulation_steps > 1:
        self.gradient_accumulator(gradients)
    self.train_loss.update_state(scaled_loss)
    if self.args.gradient_accumulation_steps == 1:
        return gradients
