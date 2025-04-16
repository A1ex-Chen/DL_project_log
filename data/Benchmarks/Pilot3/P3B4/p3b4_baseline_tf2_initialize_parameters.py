def initialize_parameters(default_model='p3b4_default_model.txt'):
    p3b4Bmk = bmk.BenchmarkP3B4(bmk.file_path, default_model, 'keras', prog
        ='p3b4_baseline', desc=
        'Hierarchical Self-Attention Network for                                 data extraction - Pilot 3 Benchmark 4'
        )
    gParameters = candle.finalize_parameters(p3b4Bmk)
    return gParameters
