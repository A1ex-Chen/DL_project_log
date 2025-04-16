def update_features(self, feat):
    """Update features vector and smooth it using exponential moving average."""
    feat /= np.linalg.norm(feat)
    self.curr_feat = feat
    if self.smooth_feat is None:
        self.smooth_feat = feat
    else:
        self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha
            ) * feat
    self.features.append(feat)
    self.smooth_feat /= np.linalg.norm(self.smooth_feat)
