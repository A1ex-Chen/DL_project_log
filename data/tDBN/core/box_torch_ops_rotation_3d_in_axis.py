def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([tstack([rot_cos, zeros, -rot_sin]), tstack([
            zeros, ones, zeros]), tstack([rot_sin, zeros, rot_cos])])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([tstack([rot_cos, -rot_sin, zeros]), tstack([
            rot_sin, rot_cos, zeros]), tstack([zeros, zeros, ones])])
    elif axis == 0:
        rot_mat_T = tstack([tstack([zeros, rot_cos, -rot_sin]), tstack([
            zeros, rot_sin, rot_cos]), tstack([ones, zeros, zeros])])
    else:
        raise ValueError('axis should in range')
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))
