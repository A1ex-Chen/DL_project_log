def test_inputs_embeds(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    input_ids = inputs_dict['input_ids']
    del inputs_dict['input_ids']
    for model_class in self.all_model_classes:
        model = model_class(config)
        model.eval()
        wte = model.get_input_embeddings()
        inputs_dict['inputs_embeds'] = wte(input_ids)
        outputs = model(**inputs_dict)
