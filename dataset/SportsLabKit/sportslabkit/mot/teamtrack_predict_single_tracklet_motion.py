def predict_single_tracklet_motion(self, tracklet):
    y = self.motion_model(tracklet).squeeze().numpy()
    return y
