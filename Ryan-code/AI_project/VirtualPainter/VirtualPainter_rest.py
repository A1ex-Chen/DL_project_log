import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm



folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for idx in myList:
    image = cv2.imread(f"{folderPath}/{idx}")
    overlayList.append(image)

header = overlayList[3]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector()

brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find landmark
    detector.FindHand(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # 2. tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.FingersUp()

        # 4. If selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # Checking for the click
            if y1 < 180:
                if 295 < x1 < 405:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 530 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 810 < x1 < 930:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1075 < x1 < 1280:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If drawing Mode - Index finger are up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                brushThickness = 50

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # 將顏色轉成灰度圖片
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # 透過閥值把img轉成黑色和白色
    _, imgInv = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY_INV)
    # 再把灰度圖片轉成彩度
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # 將被畫過的地方都變成黑色(0, 0, 0)
    img = cv2.bitwise_and(img, imgInv)
    # 再與黑色相加 就變成原本的顏色 因為黑色是(0, 0, 0) 所以可以透過加法改變像素
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:180, 0:1280] = header

    # 用來將兩個視窗合併，並給予權重，也就是在合併會，會顯示多少。而最後的gamma參數是用來調整亮度的。
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)