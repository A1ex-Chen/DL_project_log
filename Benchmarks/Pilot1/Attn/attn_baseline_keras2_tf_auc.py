def tf_auc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.
        local_variables_initializer())
    return auc
