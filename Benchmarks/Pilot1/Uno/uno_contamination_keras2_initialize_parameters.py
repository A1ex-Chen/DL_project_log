def initialize_parameters(default_model='uno_defaultUQ_model.txt'):
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model,
        'keras', prog='uno_contamination', desc=
        'Build neural network based models to predict tumor response to single and paired drugs. Use contamination model for detecting outliers in training.'
        )
    unoBmk.required.update(required)
    unoBmk.additional_definitions += additional_definitions
    gParameters = candle.finalize_parameters(unoBmk)
    return gParameters
