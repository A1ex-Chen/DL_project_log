def update(self, current_frame):
    current_frame = current_frame[0]
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
    ret, self.roi = cv2.meanShift(dst, self.roi, (cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT, self.max_iterations, self.termination_eps))
    x, y, w, h = self.roi
    self.state = {'box': (x, y, w, h)}
    return [self.state]
