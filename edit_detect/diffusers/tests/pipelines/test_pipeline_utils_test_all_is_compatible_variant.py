def test_all_is_compatible_variant(self):
    filenames = ['safety_checker/pytorch_model.fp16.bin',
        'safety_checker/model.fp16.safetensors',
        'vae/diffusion_pytorch_model.fp16.bin',
        'vae/diffusion_pytorch_model.fp16.safetensors',
        'text_encoder/pytorch_model.fp16.bin',
        'text_encoder/model.fp16.safetensors',
        'unet/diffusion_pytorch_model.fp16.bin',
        'unet/diffusion_pytorch_model.fp16.safetensors']
    variant = 'fp16'
    self.assertTrue(is_safetensors_compatible(filenames, variant=variant))
