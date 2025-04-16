@tf.function
def __call__(self, data_batch, training=True, broadcast=False):
    if not training:
        model_outputs = self.model(data_batch['features'], data_batch.get(
            'labels'), training=training)
        model_outputs.update({'source_ids': data_batch['features'][
            'source_ids'], 'image_info': data_batch['features']['image_info']})
        return model_outputs
    else:
        with tf.GradientTape() as tape:
            model_outputs = self.model(*data_batch, training=True,
                weight_decay=self.weight_decay)
            if self.fp16:
                scaled_loss = self.optimizer.get_scaled_loss(model_outputs[
                    'total_loss'])
        if self.dist != None:
            tape = self.dist.DistributedGradientTape(tape)
        if self.fp16:
            scaled_gradients = tape.gradient(scaled_loss, self.model.
                trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(model_outputs['total_loss'], self.
                model.trainable_variables)
        if self.global_gradient_clip_ratio > 0.0:
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite
                (g)) for g in gradients])
            clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=
                self.global_gradient_clip_ratio, use_norm=tf.cond(
                all_are_finite, lambda : tf.linalg.global_norm(gradients), 
                lambda : tf.constant(1.0)))
            gradients = clipped_grads
        grads_and_vars = []
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is not None and any([(pattern in var.name) for pattern in
                ['bias', 'beta']]):
                grad = 2.0 * grad
            grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)
        if self.dist != None and broadcast:
            if MPI_rank() == 0:
                logging.info('Broadcasting model')
            self.dist.broadcast_variables(self.model.variables, 0)
            self.dist.broadcast_variables(self.optimizer.variables(), 0)
        losses = {i: j for i, j in model_outputs.items() if 'loss' in i}
        model_outputs.update({'source_ids': data_batch[0]['source_ids'],
            'image_info': data_batch[0]['image_info']})
        return losses, model_outputs
