def flow_warp(feature, flow, mask=False, mode='bilinear', padding_mode='zeros'
    ):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    grid = coords_grid(b, h, w).to(flow.device) + flow
    grid = grid.to(feature.dtype)
    return bilinear_sample(feature, grid, mode=mode, padding_mode=
        padding_mode, return_mask=mask)
