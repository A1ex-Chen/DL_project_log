def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255,
    0, 255)):
    """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """
    center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
    cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
    cv2.line(self.im, center_point, center_bbox, color, self.tf)
