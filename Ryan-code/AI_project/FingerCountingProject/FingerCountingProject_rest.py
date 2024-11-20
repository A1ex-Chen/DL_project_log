import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "C:\\Users\\ryan\\Documents\\github-code\\FingerCountingProject\\image"
myList = os.listdir(folderPath)
print(myList)
overLayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    # 讀取每張圖片像素
    overLayList.append(image)
    # print(imPath)
detector = htm.HandDetector(detectionCon=0.75, maxHand=1)

tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    detector.FindHand(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    left = False

    if len(lmList) != 0:
        fingers = []

        # check which hand is detected
        for id in range(1, 5):
            if lmList[tipIds[0]][1] < lmList[tipIds[id]][1]:
                left = True

        if left:
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = overLayList[totalFingers].shape
        img[0:h, 0:w] = overLayList[totalFingers]

        cv2.rectangle(img, (20, 375), (120, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (50, 420), cv2.FONT_HERSHEY_PLAIN, 4,
                    (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (450, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)