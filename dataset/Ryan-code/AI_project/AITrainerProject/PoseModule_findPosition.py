def findPosition(self, img, pose=0, draw=True):
    self.lmList = list()
    if self.result.pose_landmarks:
        for id, lm in enumerate(self.result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return self.lmList
