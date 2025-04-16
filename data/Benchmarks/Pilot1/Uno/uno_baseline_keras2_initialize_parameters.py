def initialize_parameters(default_model='uno_default_model.txt'):
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model,
        'keras', prog='uno_baseline', desc=
        'Build neural network based models to predict tumor response to single and paired drugs.'
        )
    gParameters = candle.finalize_parameters(unoBmk)
    return gParameters
