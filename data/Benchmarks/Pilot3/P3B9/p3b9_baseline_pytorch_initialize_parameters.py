def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""
    p3b9_bench = bmk.BenchmarkP3B9(bmk.file_path, 'default_model.txt',
        'pytorch', prog='p3b9', desc='BERT')
    gParameters = candle.finalize_parameters(p3b9_bench)
    return gParameters
