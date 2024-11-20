def initialize_parameters(default_model='class_default_model.txt'):
    sctBmk = st.BenchmarkST(st.file_path, default_model, 'keras', prog=
        'sct_baseline', desc='Transformer model for SMILES classification')
    gParameters = candle.finalize_parameters(sctBmk)
    return gParameters
