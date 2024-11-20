def findFaceMash(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.faceMesh.process(imgRGB)
    faces = []
    if self.results.multi_face_landmarks:
        for faceLms in self.results.multi_face_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.
                    FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
            face = []
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                face.append([x, y])
            faces.append(face)
    return img, faces
