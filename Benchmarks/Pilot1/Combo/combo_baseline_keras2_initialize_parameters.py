def initialize_parameters(default_model='combo_default_model.txt'):
    comboBmk = combo.BenchmarkCombo(combo.file_path, default_model, 'keras',
        prog='combo_baseline', desc=
        'Build neural network based models to predict tumor response to drug pairs.'
        )
    gParameters = candle.finalize_parameters(comboBmk)
    return gParameters
