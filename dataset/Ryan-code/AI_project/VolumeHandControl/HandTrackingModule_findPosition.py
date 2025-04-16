def findPosition(self, img, handNo=0, draw=True):
    lmlist = []
    if self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
    return lmlist
