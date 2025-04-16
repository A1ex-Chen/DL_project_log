def initialize_parameters(default_model='nt3_noise_model.txt'):
    nt3Bmk = BenchmarkNT3Abs(bmk.file_path, default_model, 'keras', prog=
        'nt3_abstention', desc=
        '1D CNN to classify RNA sequence data in normal or tumor classes')
    gParameters = candle.finalize_parameters(nt3Bmk)
    return gParameters
