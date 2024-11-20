def collect_stats(model, data_loader, batch_number, device='cuda'):
    """Feed data to the network and collect statistic"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i, data_tuple in enumerate(data_loader):
        image = data_tuple[0]
        image = image.float() / 255.0
        model(image.to(device))
        if i + 1 >= batch_number:
            break
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
