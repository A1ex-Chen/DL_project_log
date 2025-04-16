def initialize_parameters(default_model='p3b3_default_model.txt'):
    p3b3Bmk = bmk.BenchmarkP3B3(bmk.file_path, default_model, 'keras', prog
        ='p3b3_baseline', desc=
        'Multi-task CNN for data extraction from clinical reports - Pilot 3 Benchmark 3'
        )
    gParameters = candle.finalize_parameters(p3b3Bmk)
    return gParameters
