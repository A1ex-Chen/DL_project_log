def display_frames(self, im0):
    """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
    if self.env_check:
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, im0)
        if cv2.waitKey(1) & 255 == ord('q'):
            return
