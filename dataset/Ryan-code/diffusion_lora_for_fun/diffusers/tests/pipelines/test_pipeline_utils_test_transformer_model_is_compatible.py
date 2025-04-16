def test_transformer_model_is_compatible(self):
    filenames = ['text_encoder/pytorch_model.bin',
        'text_encoder/model.safetensors']
    self.assertTrue(is_safetensors_compatible(filenames))
