def test_diffusers_model_is_compatible_variant(self):
    filenames = ['unet/diffusion_pytorch_model.fp16.bin',
        'unet/diffusion_pytorch_model.fp16.safetensors']
    variant = 'fp16'
    self.assertTrue(is_safetensors_compatible(filenames, variant=variant))
