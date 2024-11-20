def initialize_parameters(default_model='adrp_default_model.txt'):
    adrpBmk = adrp.BenchmarkAdrp(adrp.file_path, default_model, 'keras',
        prog='adrp_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(adrpBmk)
    return gParameters
