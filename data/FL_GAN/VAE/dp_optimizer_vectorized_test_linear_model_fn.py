def linear_model_fn(features, labels, mode):
    preds = tf.keras.layers.Dense(1, activation='linear', name='dense')(
        features['x'])
    vector_loss = tf.math.squared_difference(labels, preds)
    scalar_loss = tf.reduce_mean(input_tensor=vector_loss)
    optimizer = VectorizedDPSGD(l2_norm_clip=1.0, noise_multiplier=0.0,
        num_microbatches=1, learning_rate=1.0)
    global_step = tf.compat.v1.train.get_global_step()
    train_op = optimizer.minimize(loss=vector_loss, global_step=global_step)
    return tf_estimator.EstimatorSpec(mode=mode, loss=scalar_loss, train_op
        =train_op)
