def point_wise_feed_forward_network(d_model_size, dff, name=''):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=
        'relu', name='0'), tf.keras.layers.Dense(d_model_size, name='2')],
        name='ffn')
