def prediction_step(self, features: tf.Tensor, labels: tf.Tensor,
    nb_instances_in_global_batch: tf.Tensor) ->tf.Tensor:
    """
        Compute the prediction on features and update the loss with labels.

        Subclass and override to inject some custom behavior.
        """
    per_example_loss, logits = self.run_model(features, labels, False)
    scaled_loss = per_example_loss / tf.cast(nb_instances_in_global_batch,
        dtype=per_example_loss.dtype)
    self.eval_loss.update_state(scaled_loss)
    return logits
