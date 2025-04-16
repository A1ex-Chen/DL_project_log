def warp_single_latent(latent, reference_flow):
    """
    Warp latent of a single frame with given flow

    Args:
        latent: latent code of a single frame
        reference_flow: flow which to warp the latent with

    Returns:
        warped: warped latent
    """
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype)
    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode='bilinear')
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1))
    warped = grid_sample(latent, coords_t0, mode='nearest', padding_mode=
        'reflection')
    return warped
