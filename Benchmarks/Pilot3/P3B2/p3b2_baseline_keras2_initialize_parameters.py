def initialize_parameters(default_model='p3b2_default_model.txt'):
    p3b2Bmk = bmk.BenchmarkP3B2(bmk.file_path, default_model, 'keras', prog
        ='p3b2_baseline', desc=
        'Multi-task (DNN) for data extraction from                                 clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(p3b2Bmk)
    return gParameters
