def display_frames(self):
    """Displays the current frame."""
    cv2.imshow('Ultralytics Speed Estimation', self.im0)
    if cv2.waitKey(1) & 255 == ord('q'):
        return
