def initialize_parameters(default_model='regress_default_model.txt'):
    sctBmk = st.BenchmarkST(st.file_path, default_model, 'keras', prog=
        'p1b1_baseline', desc=
        'Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1'
        )
    gParameters = candle.finalize_parameters(sctBmk)
    return gParameters
