def test_model_common_attributes(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)
        x = model.get_output_embeddings()
        assert x is None or isinstance(x, tf.keras.layers.Layer)
