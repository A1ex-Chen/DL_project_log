def camera_to_lidar(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)],
        dim=-1)
    lidar_points = points @ torch.inverse((r_rect @ velo2cam).t())
    return lidar_points[..., :3]
