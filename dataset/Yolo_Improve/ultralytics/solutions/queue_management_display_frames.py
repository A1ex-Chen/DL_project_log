def display_frames(self):
    """Displays the current frame with annotations."""
    if self.env_check:
        self.annotator.draw_region(reg_pts=self.reg_pts, thickness=self.
            region_thickness, color=self.region_color)
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.im0)
        if cv2.waitKey(1) & 255 == ord('q'):
            return
