def test_from_pretrained_hub(self):
    model, loading_info = AutoencoderKL.from_pretrained(
        'fusing/autoencoder-kl-dummy', output_loading_info=True)
    self.assertIsNotNone(model)
    self.assertEqual(len(loading_info['missing_keys']), 0)
    model.to(torch_device)
    image = model(**self.dummy_input)
    assert image is not None, 'Make sure output is not None'
