def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""
    p3b6_bench = bmk.BenchmarkP3B6(bmk.file_path, 'default_model.txt',
        'pytorch', prog='p3b6', desc='BERT bench')
    gParameters = candle.finalize_parameters(p3b6_bench)
    return gParameters
