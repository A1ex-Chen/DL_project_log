def test_transformer_model_is_compatible_variant(self):
    filenames = ['text_encoder/pytorch_model.fp16.bin',
        'text_encoder/model.fp16.safetensors']
    variant = 'fp16'
    self.assertTrue(is_safetensors_compatible(filenames, variant=variant))
