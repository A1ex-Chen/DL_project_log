def update(self, frame):
    self.video.write(frame)
    cv2.waitKey(1)
    if self.frame_number > self.length:
        cv2.destroyAllWindows()
        self.video.release()
