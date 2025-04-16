def token_type_ids_to_mat(self, token_type_ids):
    """Convert `token_type_ids` to `token_type_mat`."""
    token_type_mat = tf.equal(tf.expand_dims(token_type_ids, -1), tf.
        expand_dims(token_type_ids, -2))
    cls_ids = tf.equal(token_type_ids, tf.constant([self.cls_token_type_id],
        dtype=token_type_ids.dtype))
    cls_mat = tf.logical_or(tf.expand_dims(cls_ids, -1), tf.expand_dims(
        cls_ids, -2))
    return tf.logical_or(cls_mat, token_type_mat)
