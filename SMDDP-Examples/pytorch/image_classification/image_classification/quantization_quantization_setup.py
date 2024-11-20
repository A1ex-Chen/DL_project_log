def quantization_setup(calib_method='histogram'):
    """Change network into quantized version "automatically" and selects histogram as default quantization method"""
    select_default_calib_method(calib_method)
    initialize()
