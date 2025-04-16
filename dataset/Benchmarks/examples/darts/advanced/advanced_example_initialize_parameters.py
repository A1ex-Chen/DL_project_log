def initialize_parameters():
    """Initialize the parameters for the Advanced example"""
    uno_example = bmk.AdvancedExample(bmk.file_path, 'default_model.txt',
        'pytorch', prog='advanced_example', desc=
        'Differentiable Architecture Search - Advanced example')
    gParameters = candle.finalize_parameters(uno_example)
    return gParameters
