def update(self, detection_points_flatten, R=None, H=None):
    if H is not None:
        diagonal = np.diagonal(H).reshape((self.dim_z, 1))
        one_minus_diagonal = 1 - diagonal
    else:
        diagonal = np.ones((self.dim_z, 1))
        one_minus_diagonal = np.zeros((self.dim_z, 1))
    if R is not None:
        kalman_r = np.diagonal(R).reshape((self.dim_z, 1))
    else:
        kalman_r = self.default_r
    error = np.multiply(detection_points_flatten - self.x[:self.dim_z],
        diagonal)
    vel_var_plus_pos_vel_cov = self.pos_vel_covariance + self.vel_variance
    added_variances = (self.pos_variance + self.pos_vel_covariance +
        vel_var_plus_pos_vel_cov + self.q_Q + kalman_r)
    kalman_r_over_added_variances = np.divide(kalman_r, added_variances)
    vel_var_plus_pos_vel_cov_over_added_variances = np.divide(
        vel_var_plus_pos_vel_cov, added_variances)
    added_variances_or_kalman_r = np.multiply(added_variances,
        one_minus_diagonal) + np.multiply(kalman_r, diagonal)
    self.x[:self.dim_z] += np.multiply(diagonal, np.multiply(1 -
        kalman_r_over_added_variances, error))
    self.x[self.dim_z:] += np.multiply(diagonal, np.multiply(
        vel_var_plus_pos_vel_cov_over_added_variances, error))
    self.pos_variance = np.multiply(1 - kalman_r_over_added_variances,
        added_variances_or_kalman_r)
    self.pos_vel_covariance = np.multiply(
        vel_var_plus_pos_vel_cov_over_added_variances,
        added_variances_or_kalman_r)
    self.vel_variance += self.q_Q - np.multiply(diagonal, np.multiply(np.
        square(vel_var_plus_pos_vel_cov_over_added_variances), added_variances)
        )
