def get_dets_from_frame(self, frame_number):
    """this function returns a list of norfair Detections class, corresponding to frame=frame_number"""
    indexes = np.argwhere(self.matrix_detections[:, 0] == frame_number)
    detections = []
    if len(indexes) > 0:
        actual_det = self.matrix_detections[indexes]
        actual_det.shape = [actual_det.shape[0], actual_det.shape[2]]
        for det in actual_det:
            points = np.array([[det[2], det[3]], [det[4], det[5]]])
            conf = det[6]
            new_detection = Detection(points, np.array([conf, conf]))
            detections.append(new_detection)
    self.actual_detections = detections
    return detections
