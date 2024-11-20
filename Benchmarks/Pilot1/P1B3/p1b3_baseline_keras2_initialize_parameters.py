def initialize_parameters(default_model='p1b3_default_model.txt'):
    p1b3Bmk = benchmark.BenchmarkP1B3(benchmark.file_path, default_model,
        'keras', prog='p1b3_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(p1b3Bmk)
    return gParameters
