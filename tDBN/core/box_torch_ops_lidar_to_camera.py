def lidar_to_camera(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).type_as(points)],
        dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]
