def test_transformer_model_is_not_compatible(self):
    filenames = ['safety_checker/pytorch_model.bin',
        'safety_checker/model.safetensors',
        'vae/diffusion_pytorch_model.bin',
        'vae/diffusion_pytorch_model.safetensors',
        'text_encoder/pytorch_model.bin',
        'unet/diffusion_pytorch_model.bin',
        'unet/diffusion_pytorch_model.safetensors']
    self.assertFalse(is_safetensors_compatible(filenames))
