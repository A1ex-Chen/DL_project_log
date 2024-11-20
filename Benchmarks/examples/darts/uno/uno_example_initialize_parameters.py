def initialize_parameters():
    """Initialize the parameters for the Uno example"""
    uno_example = bmk.UnoExample(bmk.file_path, 'default_model.txt',
        'pytorch', prog='uno_example', desc=
        'Differentiable Architecture Search - Uno example')
    gParameters = candle.finalize_parameters(uno_example)
    return gParameters
