def test_diffusers_model_is_not_compatible_variant(self):
    filenames = ['safety_checker/pytorch_model.fp16.bin',
        'safety_checker/model.fp16.safetensors',
        'vae/diffusion_pytorch_model.fp16.bin',
        'vae/diffusion_pytorch_model.fp16.safetensors',
        'text_encoder/pytorch_model.fp16.bin',
        'text_encoder/model.fp16.safetensors',
        'unet/diffusion_pytorch_model.fp16.bin']
    variant = 'fp16'
    self.assertFalse(is_safetensors_compatible(filenames, variant=variant))