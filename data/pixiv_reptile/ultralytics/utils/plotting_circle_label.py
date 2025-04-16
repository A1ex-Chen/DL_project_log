def circle_label(self, box, label='', color=(128, 128, 128), txt_color=(255,
    255, 255), margin=2):
    """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (R, G, B).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
    if len(label) > 3:
        print(
            f'Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!'
            )
        label = label[:3]
    x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.
        sf - 0.15, self.tf)[0]
    required_radius = int((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5 / 2
        ) + margin
    cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
    text_x = x_center - text_size[0] // 2
    text_y = y_center + text_size[1] // 2
    cv2.putText(self.im, str(label), (text_x, text_y), cv2.
        FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.get_txt_color(color,
        txt_color), self.tf, lineType=cv2.LINE_AA)
