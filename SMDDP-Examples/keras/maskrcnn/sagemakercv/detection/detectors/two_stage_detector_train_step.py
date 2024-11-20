@tf.function
def train_step(self, data_batch):
    features, labels = data_batch
    with tf.GradientTape() as tape:
        model_outputs = self(features, labels, training=True, weight_decay=
            self.weight_decay)
    gradients = tape.gradient(model_outputs['total_loss'], self.
        trainable_variables)
    if self.global_gradient_clip_ratio > 0.0:
        all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for
            g in gradients])
        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=self
            .global_gradient_clip_ratio, use_norm=tf.cond(all_are_finite, 
            lambda : tf.linalg.global_norm(gradients), lambda : tf.constant
            (1.0)))
        gradients = clipped_grads
    grads_and_vars = []
    for grad, var in zip(gradients, self.trainable_variables):
        if grad is not None and any([(pattern in var.name) for pattern in [
            'bias', 'beta']]):
            grad = 2.0 * grad
        grads_and_vars.append((grad, var))
    self.optimizer.apply_gradients(grads_and_vars)
    losses = {i: j for i, j in model_outputs.items() if 'loss' in i}
    model_outputs.update({'source_ids': data_batch[0]['source_ids'],
        'image_info': data_batch[0]['image_info']})
    return losses
