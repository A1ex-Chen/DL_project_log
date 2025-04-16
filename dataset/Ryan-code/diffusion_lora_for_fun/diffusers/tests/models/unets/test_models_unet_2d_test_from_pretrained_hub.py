@slow
def test_from_pretrained_hub(self):
    model, loading_info = UNet2DModel.from_pretrained(
        'google/ncsnpp-celebahq-256', output_loading_info=True)
    self.assertIsNotNone(model)
    self.assertEqual(len(loading_info['missing_keys']), 0)
    model.to(torch_device)
    inputs = self.dummy_input
    noise = floats_tensor((4, 3) + (256, 256)).to(torch_device)
    inputs['sample'] = noise
    image = model(**inputs)
    assert image is not None, 'Make sure output is not None'
