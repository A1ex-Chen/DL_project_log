def __init__(self, R: float=4.0, Q: float=0.1, pos_variance: float=10,
    pos_vel_covariance: float=0, vel_variance: float=1):
    self.R = R
    self.Q = Q
    self.pos_variance = pos_variance
    self.pos_vel_covariance = pos_vel_covariance
    self.vel_variance = vel_variance
