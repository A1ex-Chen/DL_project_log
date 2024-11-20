def test_head_pruning_save_load_from_config_init(self):
    if not self.test_pruning:
        return
    for model_class in self.all_model_classes:
        config, inputs_dict = (self.model_tester.
            prepare_config_and_inputs_for_common())
        if 'head_mask' in inputs_dict:
            del inputs_dict['head_mask']
        config.output_attentions = True
        config.output_hidden_states = False
        heads_to_prune = {(0): list(range(1, self.model_tester.
            num_attention_heads)), (-1): [0]}
        config.pruned_heads = heads_to_prune
        model = model_class(config=config)
        model.eval()
        outputs = model(**inputs_dict)
        attentions = outputs[-1]
        self.assertEqual(attentions[0].shape[-3], 1)
        self.assertEqual(attentions[1].shape[-3], self.model_tester.
            num_attention_heads)
        self.assertEqual(attentions[-1].shape[-3], self.model_tester.
            num_attention_heads - 1)
