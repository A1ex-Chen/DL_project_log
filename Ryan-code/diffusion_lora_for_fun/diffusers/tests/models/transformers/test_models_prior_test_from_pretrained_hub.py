def test_from_pretrained_hub(self):
    model, loading_info = PriorTransformer.from_pretrained(
        'hf-internal-testing/prior-dummy', output_loading_info=True)
    self.assertIsNotNone(model)
    self.assertEqual(len(loading_info['missing_keys']), 0)
    model.to(torch_device)
    hidden_states = model(**self.dummy_input)[0]
    assert hidden_states is not None, 'Make sure output is not None'
