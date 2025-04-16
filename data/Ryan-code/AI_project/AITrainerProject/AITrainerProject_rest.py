import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("pose.mp4")

detector = pm.PoseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (480, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        angel = detector.findAngel(img, 11, 13, 15)
        per = np.interp(angel, (245, 290), (0, 100))
        bar = np.interp(angel, (245, 290), (700, 500))
        # print(angel, per)

        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw bar
        cv2.rectangle(img, (20, 500), (50, 700), color, 3)
        cv2.rectangle(img, (20, int(bar)), (50, 700), color, cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (70, 690), cv2.FONT_HERSHEY_PLAIN,
                    3, color, 2)

        #Draw crul count
        cv2.rectangle(img, (400, 0), (480, 100), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(count)}", (420, 80), cv2.FONT_HERSHEY_PLAIN,
                    5, (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (25, 70), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
