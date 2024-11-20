def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
    tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
    tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name,
        initializer=tf.zeros_initializer())
    session.run(tf.variables_initializer([tf_var]))
    session.run(tf_var)
    return tf_var
