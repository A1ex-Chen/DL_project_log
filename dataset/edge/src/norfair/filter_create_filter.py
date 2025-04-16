def create_filter(self, initial_detection: np.ndarray):
    num_points = initial_detection.shape[0]
    dim_points = initial_detection.shape[1]
    dim_z = dim_points * num_points
    dim_x = 2 * dim_z
    custom_filter = OptimizedKalmanFilter(dim_x, dim_z, pos_variance=self.
        pos_variance, pos_vel_covariance=self.pos_vel_covariance,
        vel_variance=self.vel_variance, q=self.Q, r=self.R)
    custom_filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T
    return custom_filter
