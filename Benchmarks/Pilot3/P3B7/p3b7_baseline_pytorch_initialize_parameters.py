def initialize_parameters():
    """Initialize the parameters for the P3B7 benchmark"""
    p3b7_bench = bmk.BenchmarkP3B7(bmk.file_path, 'default_model.txt',
        'pytorch', prog='p3b7', desc='Network pruning')
    gParameters = candle.finalize_parameters(p3b7_bench)
    return gParameters
