def test_diffusers_model_is_compatible_variant_partial(self):
    filenames = ['unet/diffusion_pytorch_model.bin',
        'unet/diffusion_pytorch_model.safetensors']
    variant = 'fp16'
    self.assertTrue(is_safetensors_compatible(filenames, variant=variant))
