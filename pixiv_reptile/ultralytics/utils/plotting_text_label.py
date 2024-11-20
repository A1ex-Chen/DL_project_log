def text_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 
    255, 255), margin=5):
    """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (R, G, B).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
    x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 
        0.1, self.tf)[0]
    text_x = x_center - text_size[0] // 2
    text_y = y_center + text_size[1] // 2
    rect_x1 = text_x - margin
    rect_y1 = text_y - text_size[1] - margin
    rect_x2 = text_x + text_size[0] + margin
    rect_y2 = text_y + margin
    cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
    cv2.putText(self.im, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        self.sf - 0.1, self.get_txt_color(color, txt_color), self.tf,
        lineType=cv2.LINE_AA)
