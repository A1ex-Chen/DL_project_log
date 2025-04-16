def compute_loss(self, labels, logits):
    """CrossEntropyLoss that ignores pad tokens"""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=
        True, reduction=tf.keras.losses.Reduction.NONE)
    melted_labels = tf.reshape(labels, (-1,))
    active_loss = tf.not_equal(melted_labels, self.config.pad_token_id)
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(
        logits)[2])), active_loss)
    labels = tf.boolean_mask(melted_labels, active_loss)
    return loss_fn(labels, reduced_logits)
