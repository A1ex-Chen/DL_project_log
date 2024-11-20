def test_diffusers_model_is_compatible(self):
    filenames = ['unet/diffusion_pytorch_model.bin',
        'unet/diffusion_pytorch_model.safetensors']
    self.assertTrue(is_safetensors_compatible(filenames))
