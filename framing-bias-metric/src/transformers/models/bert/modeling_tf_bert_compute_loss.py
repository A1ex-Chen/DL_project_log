def compute_loss(self, labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=
        True, reduction=tf.keras.losses.Reduction.NONE)
    masked_lm_active_loss = tf.not_equal(tf.reshape(labels['labels'], (-1,)
        ), -100)
    masked_lm_reduced_logits = tf.boolean_mask(tf.reshape(logits[0], (-1,
        shape_list(logits[0])[2])), masked_lm_active_loss)
    masked_lm_labels = tf.boolean_mask(tf.reshape(labels['labels'], (-1,)),
        masked_lm_active_loss)
    next_sentence_active_loss = tf.not_equal(tf.reshape(labels[
        'next_sentence_label'], (-1,)), -100)
    next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits[1], (-
        1, 2)), next_sentence_active_loss)
    next_sentence_label = tf.boolean_mask(tf.reshape(labels[
        'next_sentence_label'], (-1,)), mask=next_sentence_active_loss)
    masked_lm_loss = loss_fn(masked_lm_labels, masked_lm_reduced_logits)
    next_sentence_loss = loss_fn(next_sentence_label,
        next_sentence_reduced_logits)
    masked_lm_loss = tf.reshape(masked_lm_loss, (-1, shape_list(
        next_sentence_loss)[0]))
    masked_lm_loss = tf.reduce_mean(masked_lm_loss, 0)
    return masked_lm_loss + next_sentence_loss
