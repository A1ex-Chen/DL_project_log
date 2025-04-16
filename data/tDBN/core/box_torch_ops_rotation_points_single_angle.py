def rotation_points_single_angle(points, angle, axis=0):
    rot_sin = math.sin(angle)
    rot_cos = math.cos(angle)
    point_type = torchplus.get_tensor_class(points)
    if axis == 1:
        rot_mat_T = torch.stack([torch.tensor([rot_cos, 0, -rot_sin], dtype
            =points.dtype, device=points.device), torch.tensor([0, 1, 0],
            dtype=points.dtype, device=points.device), torch.tensor([
            rot_sin, 0, rot_cos], dtype=points.dtype, device=points.device)])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([torch.tensor([rot_cos, -rot_sin, 0], dtype
            =points.dtype, device=points.device), torch.tensor([rot_sin,
            rot_cos, 0], dtype=points.dtype, device=points.device), torch.
            tensor([0, 0, 1], dtype=points.dtype, device=points.device)])
    elif axis == 0:
        rot_mat_T = torch.stack([torch.tensor([1, 0, 0], dtype=points.dtype,
            device=points.device), torch.tensor([0, rot_cos, -rot_sin],
            dtype=points.dtype, device=points.device), torch.tensor([0,
            rot_sin, rot_cos], dtype=points.dtype, device=points.device)])
    else:
        raise ValueError('axis should in range')
    return points @ rot_mat_T
