def write_and_display(self, im0):
    """
        Write and display the line graph
        Args:
            im0 (ndarray): Image for processing
        """
    im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
    cv2.imshow(self.title, im0) if self.view_img else None
    self.writer.write(im0) if self.save_img else None
