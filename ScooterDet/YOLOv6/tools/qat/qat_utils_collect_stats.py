def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    for i, (image, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.float() / 255.0
        model(image.cuda())
        if i >= num_batches:
            break
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
