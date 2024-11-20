@tf.function
def distributed_prediction_steps(self, batch):
    nb_instances_in_batch = self._compute_nb_instances(batch)
    inputs = self._get_step_inputs(batch, nb_instances_in_batch)
    logits = self.args.strategy.run(self.prediction_step, inputs)
    return logits
