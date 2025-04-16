import mediapipe as mp
import cv2
import time


class HandDetector():
    def __init__(self, mode=False, maxHand=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHand
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # 引入MediaPipe函式庫中的手部檢測模組
        # mp.solutions.hands 是 MediaPipe 中負責手部檢測的模組。
        self.mpHands = mp.solutions.hands

        # 建立了一個 Hands 的物件實例，用於後續的手部檢測。
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)

        # drawing_utils 模組提供了一些函數，可以在圖像上繪製不同類型的標記
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]


    def FindHand(self, img, draw = True):
        # 將從攝像頭捕捉到的BGR格式的影像轉換為RGB格式。
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 使用MediaPipe手部檢測模組(hands)對轉換後的RGB影像(imaRGB)進行手部檢測。
        self.results = self.hands.process(imgRGB)

        # results.multi_hand_landmarks返回多個手指關節座標
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 把每個座標用點畫出來
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmList= []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10,
                               (255, 0, 255), cv2.FILLED)
        return self.lmList

    def FingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


# dummy code






# dummy code
def main():

    pTime = 0
    Ctime = 0
    # 創建一個視訊捕捉對象，0為主相機，1可能為外接相機等等
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        # cap.read() 返回一個元組 (success, image)
        # 其中 success 是一個布林值，表示是否成功讀取視訊幀
        # 而 img 是包含讀取視訊幀的圖像。
        success, img = cap.read()
        img = detector.FindHand(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # putText(輸出字串, 字串顯示位置, 字體, 字體大小, 文字顏色, 文字線條粗細)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # imshow函數將讀取的視訊幀顯示在名為 "Image" 的視窗中。
        cv2.imshow("Image", img)
        cv2.waitKey(2)


if __name__ == "__main__":
    main()