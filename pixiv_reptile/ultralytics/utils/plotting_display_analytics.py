def display_analytics(self, im0, text, txt_color, bg_color, margin):
    """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): inference image
            text (dict): labels dictionary
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            margin (int): gap between text and rectangle for better display
        """
    horizontal_gap = int(im0.shape[1] * 0.02)
    vertical_gap = int(im0.shape[0] * 0.01)
    text_y_offset = 0
    for label, value in text.items():
        txt = f'{label}: {value}'
        text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
        if text_size[0] < 5 or text_size[1] < 5:
            text_size = 5, 5
        text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
        text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
        rect_x1 = text_x - margin * 2
        rect_y1 = text_y - text_size[1] - margin * 2
        rect_x2 = text_x + text_size[0] + margin * 2
        rect_y2 = text_y + margin * 2
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1
            )
        cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self
            .tf, lineType=cv2.LINE_AA)
        text_y_offset = rect_y2
