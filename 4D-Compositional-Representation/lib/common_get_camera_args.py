def get_camera_args(data, loc_field=None, scale_field=None, device=None):
    """ Returns dictionary of camera arguments.

    Args:
        data (dict): data dictionary
        loc_field (str): name of location field
        scale_field (str): name of scale field
        device (device): pytorch device
    """
    Rt = data['inputs.world_mat'].to(device)
    K = data['inputs.camera_mat'].to(device)
    if loc_field is not None:
        loc = data[loc_field].to(device)
    else:
        loc = torch.zeros(K.size(0), 3, device=K.device, dtype=K.dtype)
    if scale_field is not None:
        scale = data[scale_field].to(device)
    else:
        scale = torch.zeros(K.size(0), device=K.device, dtype=K.dtype)
    Rt = fix_Rt_camera(Rt, loc, scale)
    K = fix_K_camera(K, img_size=137.0)
    kwargs = {'Rt': Rt, 'K': K}
    return kwargs
