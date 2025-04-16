def initialize_parameters(default_model='uno_defaultUQ_model.txt'):
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model,
        'keras', prog='uno_trainUQ', desc=
        'Build and train neural network based models to predict tumor response to single and paired drugs with UQ.'
        )
    unoBmk.required.update(required)
    unoBmk.additional_definitions += additional_definitions
    gParameters = candle.finalize_parameters(unoBmk)
    return gParameters
