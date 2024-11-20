def test_compile_tf_model(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    input_ids = tf.keras.Input(batch_shape=(2, 2000), name='input_ids',
        dtype='int32')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-05, epsilon=1e-08,
        clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    for model_class in self.all_model_classes:
        model = model_class(config)
        with TemporaryDirectory() as tmpdirname:
            outputs = model(inputs_dict)
            model.save_pretrained(tmpdirname)
            model = model_class.from_pretrained(tmpdirname)
        outputs_dict = model(input_ids)
        hidden_states = outputs_dict[0]
        outputs = tf.keras.layers.Dense(2, activation='softmax', name='outputs'
            )(hidden_states)
        extended_model = tf.keras.Model(inputs=[input_ids], outputs=[outputs])
        extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric]
            )
