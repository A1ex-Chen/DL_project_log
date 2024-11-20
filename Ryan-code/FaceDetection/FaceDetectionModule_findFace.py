def findFace(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.result = self.faceDetection.process(imgRGB)
    bboxs = []
    if self.result.detections:
        for id, detection in enumerate(self.result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.
                width * iw), int(bboxC.height * ih)
            bboxs.append([id, bbox, detection.score])
            if draw:
                img = self.fancyDraw(img, bbox)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox
                    [0] - 100, bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,
                    255), 2)
    return img, bboxs
