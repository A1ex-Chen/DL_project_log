def select_default_calib_method(calib_method='histogram'):
    """Set up selected calibration method in whole network"""
    quant_desc_input = QuantDescriptor(calib_method=calib_method)
    quant_nn.QuantConv1d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(
        quant_desc_input)
