import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##################################
wCam, hCam = 640, 480
##################################

cap = cv2.VideoCapture(0)
# id = 3 是指相機視窗寬度大小
cap.set(3, wCam)
# id = 4 是指相機視窗高度大小
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)


# 從github上面抓的 可用來控制系統的聲音大小
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
percentage = 0

while True:
    success, img = cap.read()
    img = detector.FindHand(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand Range 15 ~ 200
        # Volume Range -45.0 ~ 0.0
        # np.interp() 可幫助你直接做等比轉換 ex: 15 - 200 轉成 -45 - 0.0 的相對比例是多少
        vol = np.interp(length, [15, 200], [minVol, maxVol])
        volBar = np.interp(length, [15, 200], [400, 150])
        percentage = np.interp(length, [15, 200], [0, 100])

        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(percentage)) + '%', (50, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)