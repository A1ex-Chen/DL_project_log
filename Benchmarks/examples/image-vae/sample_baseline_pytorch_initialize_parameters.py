def initialize_parameters(default_model='sample_default_model.txt'):
    sampleBmk = BenchmarkSample(file_path, default_model, 'pytorch', prog=
        'sample_baseline', desc='PyTorch ImageNet')
    gParameters = candle.finalize_parameters(sampleBmk)
    return gParameters
