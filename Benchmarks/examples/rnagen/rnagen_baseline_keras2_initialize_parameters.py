def initialize_parameters(default_model='rnagen_default_model.txt'):
    rnagenBmk = BenchmarkRNAGen(file_path, default_model, 'keras', prog=
        'rnagen_baseline', desc='RNAseq generator')
    gParameters = candle.finalize_parameters(rnagenBmk)
    return gParameters
