def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""
    p3b5_bench = bmk.BenchmarkP3B5(bmk.file_path, 'p3b5_default_model.txt',
        'pytorch', prog='p3b5_baseline', desc=
        'Differentiable Architecture Search - Pilot 3 Benchmark 5')
    gParameters = candle.finalize_parameters(p3b5_bench)
    return gParameters
