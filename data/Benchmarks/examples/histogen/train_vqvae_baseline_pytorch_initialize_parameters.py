def initialize_parameters(default_model='train_vqvae_default_model.txt'):
    trvq = TrainBk(file_path, default_model, 'pytorch', prog=
        'train_vqae_baseline', desc='Histology train vqae - Examples')
    print('Created sample benchmark')
    gParameters = candle.finalize_parameters(trvq)
    print('Parameters initialized')
    return gParameters
