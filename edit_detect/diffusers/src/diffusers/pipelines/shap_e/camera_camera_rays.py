@property
def camera_rays(self):
    batch_size, *inner_shape = self.shape
    inner_batch_size = int(np.prod(inner_shape))
    coords = self.get_image_coords()
    coords = torch.broadcast_to(coords.unsqueeze(0), [batch_size *
        inner_batch_size, *coords.shape])
    rays = self.get_camera_rays(coords)
    rays = rays.view(batch_size, inner_batch_size * self.height * self.
        width, 2, 3)
    return rays
