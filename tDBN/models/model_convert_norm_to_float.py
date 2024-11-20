@staticmethod
def convert_norm_to_float(net):
    """
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        """
    if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
        net.float()
    for child in net.children():
        VoxelNet.convert_norm_to_float(net)
    return net
