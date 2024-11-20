def apply_gradients(self, features, labels, nb_instances_in_global_batch):
    if self.args.gradient_accumulation_steps == 1:
        gradients = self.training_step(features, labels,
            nb_instances_in_global_batch)
        self.optimizer.apply_gradients(list(zip(gradients, self.model.
            trainable_variables)))
    else:
        for _ in tf.range(self.args.gradient_accumulation_steps):
            reduced_features = {k: ft[:self.args.train_batch_size // self.
                args.n_replicas] for k, ft in features.items()}
            reduced_labels = labels[:self.args.train_batch_size // self.
                args.n_replicas]
            self.training_step(reduced_features, reduced_labels,
                nb_instances_in_global_batch)
            features = {k: tf.concat([ft[self.args.train_batch_size // self
                .args.n_replicas:], reduced_features[k]], axis=0) for k, ft in
                features.items()}
            labels = tf.concat([labels[self.args.train_batch_size // self.
                args.n_replicas:], reduced_labels], axis=0)
        gradients = self.gradient_accumulator.gradients
        gradients = [tf.clip_by_value(grad, -self.args.max_grad_norm, self.
            args.max_grad_norm) for grad in gradients]
        self.optimizer.apply_gradients(list(zip(gradients, self.model.
            trainable_variables)))
        self.gradient_accumulator.reset()
