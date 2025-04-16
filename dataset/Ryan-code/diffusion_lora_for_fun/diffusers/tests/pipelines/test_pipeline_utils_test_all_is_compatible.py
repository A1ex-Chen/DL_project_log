def test_all_is_compatible(self):
    filenames = ['safety_checker/pytorch_model.bin',
        'safety_checker/model.safetensors',
        'vae/diffusion_pytorch_model.bin',
        'vae/diffusion_pytorch_model.safetensors',
        'text_encoder/pytorch_model.bin', 'text_encoder/model.safetensors',
        'unet/diffusion_pytorch_model.bin',
        'unet/diffusion_pytorch_model.safetensors']
    self.assertTrue(is_safetensors_compatible(filenames))
