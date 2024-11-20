def test_keyword_and_dict_args(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        outputs_dict = model(inputs_dict)
        inputs_keywords = copy.deepcopy(inputs_dict)
        input_ids = inputs_keywords.pop('input_ids')
        outputs_keywords = model(input_ids, **inputs_keywords)
        output_dict = outputs_dict[0].numpy()
        output_keywords = outputs_keywords[0].numpy()
        self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-06)
