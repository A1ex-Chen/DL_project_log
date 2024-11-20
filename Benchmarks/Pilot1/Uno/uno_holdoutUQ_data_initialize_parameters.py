def initialize_parameters(default_model='uno_defaultUQ_model.txt'):
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model,
        'keras', prog='uno_holdoutUQ_data', desc=
        'Build data split for UQ analysis in the problem of prediction of tumor response to single or drug pairs.'
        )
    gParameters = candle.finalize_parameters(unoBmk)
    return gParameters
