def pre_initialize(self, initial_frame, bins=16, max_iterations=10,
    termination_eps=1):
    self.bins = bins
    self.max_iterations = max_iterations
    self.termination_eps = termination_eps
    x, y, w, h = self.target['box']
    self.roi = x, y, w, h
    roi_frame = initial_frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((
        180.0, 255.0, 255.0)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [self.bins], [0, 180])
    self.hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
