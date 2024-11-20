def __init__(self):
    """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
    ndim, dt = 4, 1.0
    self._motion_mat = np.eye(2 * ndim, 2 * ndim)
    for i in range(ndim):
        self._motion_mat[i, ndim + i] = dt
    self._update_mat = np.eye(ndim, 2 * ndim)
    self._std_weight_position = 1.0 / 20
    self._std_weight_velocity = 1.0 / 160
