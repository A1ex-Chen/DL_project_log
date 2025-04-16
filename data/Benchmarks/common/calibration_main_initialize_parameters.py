def initialize_parameters(default_model='calibration_default.txt'):
    calBmk = CalibrationApp(file_path, default_model, 'python', prog=
        'calibration_main', desc=
        'script to compute empirical calibration for UQ regression')
    gParameters = candle.finalize_parameters(calBmk)
    return gParameters
