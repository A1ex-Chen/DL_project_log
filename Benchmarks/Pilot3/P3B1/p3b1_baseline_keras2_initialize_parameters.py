def initialize_parameters(default_model='p3b1_default_model.txt'):
    p3b1Bmk = bmk.BenchmarkP3B1(bmk.file_path, default_model, 'keras', prog
        ='p3b1_baseline', desc=
        'Multi-task (DNN) for data extraction                                      from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(p3b1Bmk)
    return gParameters
