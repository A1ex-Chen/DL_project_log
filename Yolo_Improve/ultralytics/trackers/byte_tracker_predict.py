def predict(self):
    """Predicts mean and covariance using Kalman filter."""
    mean_state = self.mean.copy()
    if self.state != TrackState.Tracked:
        mean_state[7] = 0
    self.mean, self.covariance = self.kalman_filter.predict(mean_state,
        self.covariance)
