def compute_loss(self, labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=
        True, reduction=tf.keras.losses.Reduction.NONE)
    next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
    next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 
        2)), next_sentence_active_loss)
    next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)),
        next_sentence_active_loss)
    return loss_fn(next_sentence_label, next_sentence_reduced_logits)
