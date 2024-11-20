def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
    center_unnormalized = query_xyz + center_offset
    center_normalized = shift_scale_points(center_unnormalized, src_range=
        point_cloud_dims)
    return center_normalized, center_unnormalized
