def initialize_parameters(default_model='p1b1_default_model.txt'):
    p1b1Bmk = p1b1.BenchmarkP1B1(p1b1.file_path, default_model, 'keras',
        prog='p1b1_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(p1b1Bmk)
    return gParameters
