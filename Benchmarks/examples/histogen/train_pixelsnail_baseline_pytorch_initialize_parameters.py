def initialize_parameters(default_model='train_pixelsnail_default_model.txt'):
    trpsn = TrPxSnBk(file_path, default_model, 'pytorch', prog=
        'train_pixelsnail_baseline', desc=
        'Histology train pixelsnail - Examples')
    print('Created sample benchmark')
    gParameters = candle.finalize_parameters(trpsn)
    print('Parameters initialized')
    return gParameters
