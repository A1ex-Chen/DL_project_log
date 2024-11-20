def test_determinism(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        first, second = model(inputs_dict, training=False)[0], model(
            inputs_dict, training=False)[0]
        self.assertTrue(tf.math.equal(first, second).numpy().all())
