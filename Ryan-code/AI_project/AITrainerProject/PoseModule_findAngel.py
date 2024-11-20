def findAngel(self, img, p1, p2, p3, draw=True):
    x1, y1 = self.lmList[p1][1:]
    x2, y2 = self.lmList[p2][1:]
    x3, y3 = self.lmList[p3][1:]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2,
        x1 - x2))
    if angle < 0:
        angle += 360
    print(angle)
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
        cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
    return angle
