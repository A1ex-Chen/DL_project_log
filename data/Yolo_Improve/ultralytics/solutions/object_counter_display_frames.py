def display_frames(self):
    """Displays the current frame with annotations and regions in a window."""
    if self.env_check:
        cv2.namedWindow(self.window_name)
        if len(self.reg_pts) == 4:
            cv2.setMouseCallback(self.window_name, self.
                mouse_event_for_region, {'region_points': self.reg_pts})
        cv2.imshow(self.window_name, self.im0)
        if cv2.waitKey(1) & 255 == ord('q'):
            return
