def predict_multi_tracklet_motion(self, tracklets):
    with torch.no_grad():
        Y = self.motion_model(tracklets)
        Y = np.array(Y).squeeze()
    return Y
