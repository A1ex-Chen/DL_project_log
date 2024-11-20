def initialize_parameters(default_model='attn_abs_default_model.txt'):
    attnAbsBmk = BenchmarkAttnAbs(attn.file_path, default_model, 'keras',
        prog='attention_abstention', desc=
        'Attention model with abstention - Pilot 1 Benchmark')
    gParameters = candle.finalize_parameters(attnAbsBmk)
    return gParameters
