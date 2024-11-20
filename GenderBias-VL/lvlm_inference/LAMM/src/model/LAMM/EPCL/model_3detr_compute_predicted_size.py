def compute_predicted_size(self, size_normalized, point_cloud_dims):
    scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
    scene_scale = torch.clamp(scene_scale, min=0.1)
    size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
    return size_unnormalized
