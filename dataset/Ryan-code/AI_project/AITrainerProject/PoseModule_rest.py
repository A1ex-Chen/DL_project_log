import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, mode = False, complexity=1,smoothLm=True,
                 enable=False, smoothSe=True, detectionCon=0.5,
                 trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothLm = smoothLm
        self.enable = enable
        self.smoothSe = smoothSe
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothLm, self.enable,
                                     self.smoothSe, self.detectionCon, self.trackCon, )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)

        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, pose=0, draw=True):
        self.lmList = list()
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0),
                           cv2.FILLED)
        return self.lmList

    def findAngel(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
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
            # cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN,
            #             2, (255, 0, 255), 2)
        return angle





# dummy code








# dummy code
def main():
    cap = cv2.VideoCapture("C:\\Users\\ryan\\PycharmProjects\\pyProject\\1.mp4")
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)

        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN
                    , 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()