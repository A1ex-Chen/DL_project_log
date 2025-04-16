def initialize_parameters(default_model='uno_default_inferUQ_model.txt'):
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model,
        'keras', prog='uno_inferUQ', desc=
        'Read models to predict tumor response to single and paired drugs.')
    unoBmk.additional_definitions += additional_definitions_local
    unoBmk.required = unoBmk.required.union(required_local)
    gParameters = candle.finalize_parameters(unoBmk)
    return gParameters
