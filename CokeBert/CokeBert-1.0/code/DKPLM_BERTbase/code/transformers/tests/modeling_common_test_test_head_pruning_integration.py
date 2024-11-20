def test_head_pruning_integration(self):
    if not self.test_pruning:
        return
    for model_class in self.all_model_classes:
        config, inputs_dict = (self.model_tester.
            prepare_config_and_inputs_for_common())
        if 'head_mask' in inputs_dict:
            del inputs_dict['head_mask']
        config.output_attentions = True
        config.output_hidden_states = False
        heads_to_prune = {(0): [0], (1): [1, 2]}
        config.pruned_heads = heads_to_prune
        model = model_class(config=config)
        model.eval()
        outputs = model(**inputs_dict)
        attentions = outputs[-1]
        self.assertEqual(attentions[0].shape[-3], self.model_tester.
            num_attention_heads - 1)
        self.assertEqual(attentions[1].shape[-3], self.model_tester.
            num_attention_heads - 2)
        self.assertEqual(attentions[2].shape[-3], self.model_tester.
            num_attention_heads)
        self.assertEqual(attentions[3].shape[-3], self.model_tester.
            num_attention_heads)
        directory = 'pruned_model'
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.save_pretrained(directory)
        model = model_class.from_pretrained(directory)
        shutil.rmtree(directory)
        outputs = model(**inputs_dict)
        attentions = outputs[-1]
        self.assertEqual(attentions[0].shape[-3], self.model_tester.
            num_attention_heads - 1)
        self.assertEqual(attentions[1].shape[-3], self.model_tester.
            num_attention_heads - 2)
        self.assertEqual(attentions[2].shape[-3], self.model_tester.
            num_attention_heads)
        self.assertEqual(attentions[3].shape[-3], self.model_tester.
            num_attention_heads)
        heads_to_prune = {(0): [0], (2): [1, 2]}
        model.prune_heads(heads_to_prune)
        outputs = model(**inputs_dict)
        attentions = outputs[-1]
        self.assertEqual(attentions[0].shape[-3], self.model_tester.
            num_attention_heads - 1)
        self.assertEqual(attentions[1].shape[-3], self.model_tester.
            num_attention_heads - 2)
        self.assertEqual(attentions[2].shape[-3], self.model_tester.
            num_attention_heads - 2)
        self.assertEqual(attentions[3].shape[-3], self.model_tester.
            num_attention_heads)
        self.assertDictEqual(model.config.pruned_heads, {(0): [0], (1): [1,
            2], (2): [1, 2]})
