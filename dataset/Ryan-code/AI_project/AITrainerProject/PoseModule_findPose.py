def findPose(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.result = self.pose.process(imgRGB)
    if self.result.pose_landmarks:
        if draw:
            self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS)
    return img
