def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""
    p3b5_bench = bmk.BenchmarkP3B5(bmk.file_path, 'default_model.txt',
        'pytorch', prog='p3b6', desc='BERT bench')
    gParameters = candle.finalize_parameters(p3b6)
    return gParameters
