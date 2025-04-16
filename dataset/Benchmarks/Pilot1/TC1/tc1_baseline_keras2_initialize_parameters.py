def initialize_parameters(default_model='tc1_default_model.txt'):
    tc1Bmk = bmk.BenchmarkTC1(file_path, default_model, 'keras', prog=
        'tc1_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(tc1Bmk)
    return gParameters
