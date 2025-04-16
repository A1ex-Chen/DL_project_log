import autopy.mouse
import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy as ap

######################################
wCam, hCam = 640, 480
pTime = 0
wScr, hScr = ap.screen.size()
frameR = 100
smooThening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0
######################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(maxHand=1)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.FindHand(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and thumb
    if len(lmList) != 0:
        x1, y1 = lmList[4][1:]
        x2, y2 = lmList[8][1:]
        # 3. check which fingers are up
        fingers = detector.FingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index Fingers : Moving mode
        if fingers[0] == 0 and fingers[1] == 1:
            # 5. Covert coordinates
            x3 = np.interp(x2, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y2, (frameR, hCam - frameR - 50), (0, hScr))
            # 6. Smoother Values
            clocX = plocX + (x3 - plocX) / smooThening
            clocY = plocY + (y3 - plocY) / smooThening
            # 7. Move Mouse
            clocX = round(max(0, min(clocX, wScr - 1)))
            clocY = round(max(0, min(clocY, hScr - 1)))
            autopy.mouse.move(clocX, clocY)
            plocX, plocY = clocX, clocY
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        # 8. Both index and thumb fingers are up : clicking mode
        if fingers[0] == 1 and fingers[1] == 1:
            # 9. Find distance between fingers
            length, img, info = detector.findDistance(4, 8, img)
            # print(length)
            # 10. Click mouse if distance short
            if length < 20:
                cv2.circle(img, (info[4], info[5]), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # 12. Display
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


