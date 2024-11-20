def initialize_parameters():
    """Initialize the parameters for the P3B5 benchmark"""
    p3b8_bench = bmk.BenchmarkP3B8(bmk.file_path, 'default_model.txt',
        'pytorch', prog='p3b8', desc='BERT Quantized')
    gParameters = candle.finalize_parameters(p3b8_bench)
    return gParameters
