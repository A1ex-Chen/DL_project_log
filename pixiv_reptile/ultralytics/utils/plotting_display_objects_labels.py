def display_objects_labels(self, im0, text, txt_color, bg_color, x_center,
    y_center, margin):
    """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): inference image
            text (str): object/class name
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            x_center (float): x position center point for bounding box
            y_center (float): y position center point for bounding box
            margin (int): gap between text and rectangle for better display
        """
    text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[
        0]
    text_x = x_center - text_size[0] // 2
    text_y = y_center + text_size[1] // 2
    rect_x1 = text_x - margin
    rect_y1 = text_y - text_size[1] - margin
    rect_x2 = text_x + text_size[0] + margin
    rect_y2 = text_y + margin
    cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf,
        lineType=cv2.LINE_AA)
