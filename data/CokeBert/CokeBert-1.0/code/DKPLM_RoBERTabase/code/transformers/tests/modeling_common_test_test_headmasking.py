def test_headmasking(self):
    if not self.test_head_masking:
        return
    global_rng.seed(42)
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    global_rng.seed()
    config.output_attentions = True
    config.output_hidden_states = True
    configs_no_init = _config_zero_init(config)
    for model_class in self.all_model_classes:
        model = model_class(config=configs_no_init)
        model.eval()
        head_mask = torch.ones(self.model_tester.num_hidden_layers, self.
            model_tester.num_attention_heads)
        head_mask[0, 0] = 0
        head_mask[-1, :-1] = 0
        head_mask.requires_grad_(requires_grad=True)
        inputs = inputs_dict.copy()
        inputs['head_mask'] = head_mask
        outputs = model(**inputs)
        output = sum(t.sum() for t in outputs[0])
        output = output.sum()
        output.backward()
        multihead_outputs = head_mask.grad
        attentions = outputs[-1]
        hidden_states = outputs[-2]
        for t in attentions:
            self.assertLess(torch.sum(torch.isnan(t)), t.numel() / 4)
        attentions = [t.masked_fill(torch.isnan(t), 0.0) for t in attentions]
        self.assertIsNotNone(multihead_outputs)
        self.assertEqual(len(multihead_outputs), self.model_tester.
            num_hidden_layers)
        self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().
            item(), 0.0)
        self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().
            item(), 0.0)
        self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().
            item(), 0.0)
        self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum(
            ).item(), 0.0)
        self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().
            item(), 0.0)
