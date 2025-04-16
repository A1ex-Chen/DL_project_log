import cv2
import mediapipe as mp
import time


class FaceMashDetector():

    def __init__(self, staticMode=False, maxFaces=2, refinelm=False, minDetectorCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refinelm = refinelm
        self.minDetectorCon = minDetectorCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMash = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMash.FaceMesh(self.staticMode, self.maxFaces, self.refinelm,
                                                 self.minDetectorCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMash(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y),
                    #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([x, y])

                faces.append(face)
        return img, faces





def main():
    cap = cv2.VideoCapture("face.mp4")
    pTime = 0
    detector = FaceMashDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMash(img, True)

        if len(faces) != 0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()