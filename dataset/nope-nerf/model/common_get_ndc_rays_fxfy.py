def get_ndc_rays_fxfy(fxfy, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]
    o0 = -1.0 / (1 / fxfy[0]) * ox_oz
    o1 = -1.0 / (1 / fxfy[1]) * oy_oz
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]
    d0 = -1.0 / (1 / fxfy[0]) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1.0 / (1 / fxfy[1]) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d
