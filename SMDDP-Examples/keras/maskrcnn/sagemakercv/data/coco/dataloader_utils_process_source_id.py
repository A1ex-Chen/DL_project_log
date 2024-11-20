def process_source_id(source_id):
    """Processes source_id to the right format."""
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int64)
    with tf.control_dependencies([source_id]):
        source_id = tf.cond(tf.equal(tf.size(source_id), 0), lambda : tf.
            cast(tf.constant(-1), tf.int64), lambda : tf.identity(source_id))
    return source_id
