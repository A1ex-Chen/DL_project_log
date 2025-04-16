def test_transformer_model_is_compatible_variant_partial(self):
    filenames = ['text_encoder/pytorch_model.bin',
        'text_encoder/model.safetensors']
    variant = 'fp16'
    self.assertTrue(is_safetensors_compatible(filenames, variant=variant))
