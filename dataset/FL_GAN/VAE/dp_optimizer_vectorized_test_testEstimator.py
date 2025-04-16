def testEstimator(self):
    """Tests that DP optimizers work with tf.estimator."""

    def linear_model_fn(features, labels, mode):
        preds = tf.keras.layers.Dense(1, activation='linear', name='dense')(
            features['x'])
        vector_loss = tf.math.squared_difference(labels, preds)
        scalar_loss = tf.reduce_mean(input_tensor=vector_loss)
        optimizer = VectorizedDPSGD(l2_norm_clip=1.0, noise_multiplier=0.0,
            num_microbatches=1, learning_rate=1.0)
        global_step = tf.compat.v1.train.get_global_step()
        train_op = optimizer.minimize(loss=vector_loss, global_step=global_step
            )
        return tf_estimator.EstimatorSpec(mode=mode, loss=scalar_loss,
            train_op=train_op)
    linear_regressor = tf_estimator.Estimator(model_fn=linear_model_fn)
    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = 6.0
    train_data = np.random.normal(scale=3.0, size=(200, 4)).astype(np.float32)
    train_labels = np.matmul(train_data, true_weights
        ) + true_bias + np.random.normal(scale=0.1, size=(200, 1)).astype(np
        .float32)
    train_input_fn = tf_compat_v1_estimator.inputs.numpy_input_fn(x={'x':
        train_data}, y=train_labels, batch_size=20, num_epochs=10, shuffle=True
        )
    linear_regressor.train(input_fn=train_input_fn, steps=100)
    self.assertAllClose(linear_regressor.get_variable_value('dense/kernel'),
        true_weights, atol=1.0)
