def initialize_parameters(default_model='nt3_default_model.txt'):
    import os
    nt3Bmk = bmk.BenchmarkNT3(bmk.file_path, os.getenv(
        'CANDLE_DEFAULT_MODEL_FILE'), 'keras', prog='nt3_baseline', desc=
        '1D CNN to classify RNA sequence data in normal or tumor classes')
    gParameters = candle.finalize_parameters(nt3Bmk)
    return gParameters
