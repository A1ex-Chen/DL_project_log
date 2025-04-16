def queue_counts_display(self, label, points=None, region_color=(255, 255, 
    255), txt_color=(0, 0, 0)):
    """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): queue counts label
            points (tuple): region points for center point calculation to display text
            region_color (RGB): queue region color
            txt_color (RGB): text display color
        """
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    center_x = sum(x_values) // len(points)
    center_y = sum(y_values) // len(points)
    text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf
        )[0]
    text_width = text_size[0]
    text_height = text_size[1]
    rect_width = text_width + 20
    rect_height = text_height + 20
    rect_top_left = center_x - rect_width // 2, center_y - rect_height // 2
    rect_bottom_right = center_x + rect_width // 2, center_y + rect_height // 2
    cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2
    cv2.putText(self.im, label, (text_x, text_y), 0, fontScale=self.sf,
        color=txt_color, thickness=self.tf, lineType=cv2.LINE_AA)
