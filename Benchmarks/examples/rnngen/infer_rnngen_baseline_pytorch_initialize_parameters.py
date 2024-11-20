def initialize_parameters(default_model='infer_rnngen_default_model.txt'):
    sample = InferBk(file_path, default_model, 'pytorch', prog=
        'infer_rnngen_baseline', desc='rnngen infer - Examples')
    print('Created sample benchmark')
    gParameters = candle.finalize_parameters(sample)
    print('Parameters initialized')
    return gParameters
