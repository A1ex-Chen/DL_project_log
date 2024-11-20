def ray_marching(self, ray0, ray_direction, model, c=None, tau=0.5, n_steps
    =[128, 129], n_secant_steps=8, depth_range=[0.0, 2.4], max_points=
    3500000, rad=1.0):
    """ Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        """
    batch_size, n_pts, D = ray0.shape
    device = ray0.device
    tau = 0.5
    n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()
    depth_intersect, _ = get_sphere_intersection(ray0[:, 0], ray_direction,
        r=rad)
    d_intersect = depth_intersect[..., 1]
    d_proposal = torch.linspace(0, 1, steps=n_steps).view(1, 1, n_steps, 1).to(
        device)
    d_proposal = depth_range[0] * (1.0 - d_proposal) + d_intersect.view(1, 
        -1, 1, 1) * d_proposal
    p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1
        ) + ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal
    with torch.no_grad():
        val = torch.cat([(self.model(p_split, only_occupancy=True) - tau) for
            p_split in torch.split(p_proposal.reshape(batch_size, -1, 3),
            int(max_points / batch_size), dim=1)], dim=1).view(batch_size, 
            -1, n_steps)
    mask_0_not_occupied = val[:, :, 0] < 0
    sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
        torch.ones(batch_size, n_pts, 1).to(device)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_steps, 0, -1).float().to(device)
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1), torch.
        arange(n_pts).unsqueeze(-0), indices] < 0
    mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied
    n = batch_size * n_pts
    d_low = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)
        ].view(batch_size, n_pts)[mask]
    f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
        batch_size, n_pts)[mask]
    indices = torch.clamp(indices + 1, max=n_steps - 1)
    d_high = d_proposal.view(n, n_steps, 1)[torch.arange(n), indices.view(n)
        ].view(batch_size, n_pts)[mask]
    f_high = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
        batch_size, n_pts)[mask]
    ray0_masked = ray0[mask]
    ray_direction_masked = ray_direction[mask]
    if c is not None and c.shape[-1] != 0:
        c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
    d_pred_out = torch.ones(batch_size, n_pts).to(device)
    if ray0_masked.shape[0] != 0:
        d_pred = self.secant(f_low, f_high, d_low, d_high, n_secant_steps,
            ray0_masked, ray_direction_masked, tau)
        d_pred_out[mask] = d_pred
    d_pred_out[mask == 0] = np.inf
    d_pred_out[mask_0_not_occupied == 0] = 0
    return d_pred_out
