def initialize_parameters(default_model='image_vae_default_model.txt'):
    image_vaeBmk = BenchmarkImageVAE(file_path, default_model, 'pytorch',
        prog='image_vae_baseline', desc='PyTorch ImageNet Training')
    gParameters = candle.finalize_parameters(image_vaeBmk)
    return gParameters
