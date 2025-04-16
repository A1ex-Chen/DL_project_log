def get_sphere_intersection(cam_loc, ray_directions, r=1.0):
    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1) ** 2 - r ** 2)
    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[
        mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[
        mask_intersect].unsqueeze(-1)
    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)
    return sphere_intersections, mask_intersect
