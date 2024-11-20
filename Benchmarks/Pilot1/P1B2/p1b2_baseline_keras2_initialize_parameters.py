def initialize_parameters(default_model='p1b2_default_model.txt'):
    p1b2Bmk = p1b2.BenchmarkP1B2(p1b2.file_path, default_model, 'keras',
        prog='p1b2_baseline', desc='Train Classifier - Pilot 1 Benchmark 2')
    gParameters = candle.finalize_parameters(p1b2Bmk)
    return gParameters
