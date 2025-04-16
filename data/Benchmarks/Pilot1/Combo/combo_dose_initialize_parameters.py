def initialize_parameters():
    comboBmk = combo.BenchmarkCombo(combo.file_path,
        'combo_default_model.txt', 'keras', prog='combo_baseline', desc=
        'Build neural network based models to predict tumor response to drug pairs.'
        )
    gParameters = candle.finalize_parameters(comboBmk)
    return gParameters
