def compute_amax(model, **kwargs):
    """Loads statistics of data and calculates quantization parameters in whole network"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer
            ) and module._calibrator is not None:
            if isinstance(module._calibrator, calib.MaxCalibrator):
                module.load_calib_amax()
            else:
                module.load_calib_amax(**kwargs)
    model.cuda()
