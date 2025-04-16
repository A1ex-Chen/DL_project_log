def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.

    Retained for legacy purposes. It is recommended to use FP16Model.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))
