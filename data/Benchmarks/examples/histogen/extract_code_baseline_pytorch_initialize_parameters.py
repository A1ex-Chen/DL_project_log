def initialize_parameters(default_model='extract_code_default_model.txt'):
    excd = ExtractCodeBk(file_path, default_model, 'pytorch', prog=
        'extract_code_baseline', desc='Histology Extract Code - Examples')
    print('Created sample benchmark')
    gParameters = candle.finalize_parameters(excd)
    print('Parameters initialized')
    return gParameters
