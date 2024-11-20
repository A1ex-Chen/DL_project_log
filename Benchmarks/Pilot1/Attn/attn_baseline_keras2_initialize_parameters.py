def initialize_parameters(default_model='attn_default_model.txt'):
    attnBmk = attn.BenchmarkAttn(attn.file_path, default_model, 'keras',
        prog='attn_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(attnBmk)
    return gParameters
