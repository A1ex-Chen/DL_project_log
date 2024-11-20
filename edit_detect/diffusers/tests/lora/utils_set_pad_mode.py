def set_pad_mode(network, mode='circular'):
    for _, module in network.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = mode
