def initialize_parameters(default_model='sample_default_model.txt'):
    sample = SampleBk(file_path, default_model, 'pytorch', prog=
        'sample_baseline', desc='Histology Sample - Examples')
    print('Created sample benchmark')
    gParameters = candle.finalize_parameters(sample)
    print('Parameters initialized')
    return gParameters
