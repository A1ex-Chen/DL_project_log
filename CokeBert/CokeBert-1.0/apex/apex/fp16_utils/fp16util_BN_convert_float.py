def BN_convert_float(module):
    """
    Utility function for network_to_half().

    Retained for legacy purposes.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm
        ) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module
