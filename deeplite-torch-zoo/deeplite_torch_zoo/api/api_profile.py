def profile(model, img_size=224, in_ch=3, fuse=True, eval_mode=True,
    include_frozen_params=False):
    """
    Do model profiling to calculate the MAC count, the number of parameters and weight size of the model.
    The torchprofile package is used to calculate MACs, which is jit-based and
    is more accurate than the hook based approach.

    :param model: PyTorch nn.Module object
    :param img_size: Input image resolution, either an integer value or a tuple of integers of the form (3, 224, 224)
    :param verbose: if True, prints the model summary table from torchinfo

    returns a dictionary with the number of GMACs, number of parameters in millions and model size in megabytes
    """
    device = next(model.parameters()).device
    if not isinstance(img_size, (int, tuple)):
        raise ValueError(
            'img_size has be either an integer (e.g. 224) or a tuple of integers (e.g. (3, 224, 224))'
            )
    if not isinstance(img_size, tuple):
        img_size = in_ch, img_size, img_size
    if fuse and hasattr(model, 'fuse'):
        model = model.fuse()
    with switch_train_mode(model, is_training=not eval_mode):
        num_params = sum(p.numel() for p in model.parameters() if 
            include_frozen_params or p.requires_grad)
        macs = profile_macs(model, torch.randn(1, *img_size).to(device))
        peak_ram = profile_ram(model, torch.randn(1, *img_size).to(device))
    return {'GMACs': macs / 1000000000.0, 'Mparams': num_params / 1000000.0,
        'PeakRAM': peak_ram}
