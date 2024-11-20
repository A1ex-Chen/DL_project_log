def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    EPS = 1e-06
    corners1 = corners1.clone()
    corners2 = corners2.clone()
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1
    al_xmin = torch.min(torch.min(corners1[:, :, :, 0], dim=2).values[:, :,
        None], torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :])
    al_ymin = torch.max(torch.max(corners1[:, :, :, 1], dim=2).values[:, :,
        None], torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :])
    al_zmin = torch.min(torch.min(corners1[:, :, :, 2], dim=2).values[:, :,
        None], torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :])
    al_xmax = torch.max(torch.max(corners1[:, :, :, 0], dim=2).values[:, :,
        None], torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :])
    al_ymax = torch.min(torch.min(corners1[:, :, :, 1], dim=2).values[:, :,
        None], torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :])
    al_zmax = torch.max(torch.max(corners1[:, :, :, 2], dim=2).values[:, :,
        None], torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :])
    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol
