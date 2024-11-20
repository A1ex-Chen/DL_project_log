def get_camera_rays(self, coords: torch.Tensor) ->torch.Tensor:
    batch_size, *shape, n_coords = coords.shape
    assert n_coords == 2
    assert batch_size == self.origin.shape[0]
    flat = coords.view(batch_size, -1, 2)
    res = self.resolution()
    fov = self.fov()
    fracs = flat.float() / (res - 1) * 2 - 1
    fracs = fracs * torch.tan(fov / 2)
    fracs = fracs.view(batch_size, -1, 2)
    directions = self.z.view(batch_size, 1, 3) + self.x.view(batch_size, 1, 3
        ) * fracs[:, :, :1] + self.y.view(batch_size, 1, 3) * fracs[:, :, 1:]
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays = torch.stack([torch.broadcast_to(self.origin.view(batch_size, 1, 
        3), [batch_size, directions.shape[1], 3]), directions], dim=2)
    return rays.view(batch_size, *shape, 2, 3)
