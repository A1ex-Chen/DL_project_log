def display_frames(self):
    """Display frame."""
    cv2.imshow('Ultralytics Heatmap', self.im0)
    if cv2.waitKey(1) & 255 == ord('q'):
        return
