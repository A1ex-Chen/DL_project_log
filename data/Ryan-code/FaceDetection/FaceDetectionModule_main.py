def main():
    cap = cv2.VideoCapture('face.mp4')
    pTime = 0
    detector = FaceDetector(0.75)
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFace(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.
            FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(10)
