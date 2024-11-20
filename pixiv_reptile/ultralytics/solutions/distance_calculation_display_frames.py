def display_frames(self):
    """Displays the current frame with annotations."""
    cv2.namedWindow('Ultralytics Distance Estimation')
    cv2.setMouseCallback('Ultralytics Distance Estimation', self.
        mouse_event_for_distance)
    cv2.imshow('Ultralytics Distance Estimation', self.im0)
    if cv2.waitKey(1) & 255 == ord('q'):
        return
