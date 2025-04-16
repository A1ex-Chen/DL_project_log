@tf.function
def distributed_training_steps(self, batch):
    with self.args.strategy.scope():
        nb_instances_in_batch = self._compute_nb_instances(batch)
        inputs = self._get_step_inputs(batch, nb_instances_in_batch)
        self.args.strategy.run(self.apply_gradients, inputs)
