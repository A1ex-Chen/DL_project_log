def draw_rotated_box_with_label(self, rotated_box, alpha=0.5, edge_color=
    'g', line_style='-', label=None):
    """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
    cnt_x, cnt_y, w, h, angle = rotated_box
    area = w * h
    linewidth = self._default_font_size / (6 if area < 
        _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3)
    theta = angle * math.pi / 180.0
    c = math.cos(theta)
    s = math.sin(theta)
    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
    rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for 
        xx, yy in rect]
    for k in range(4):
        j = (k + 1) % 4
        self.draw_line([rotated_rect[k][0], rotated_rect[j][0]], [
            rotated_rect[k][1], rotated_rect[j][1]], color=edge_color,
            linestyle='--' if k == 1 else line_style, linewidth=linewidth)
    if label is not None:
        text_pos = rotated_rect[1]
        height_ratio = h / np.sqrt(self.output.height * self.output.width)
        label_color = self._change_color_brightness(edge_color,
            brightness_factor=0.7)
        font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2
            ) * 0.5 * self._default_font_size
        self.draw_text(label, text_pos, color=label_color, font_size=
            font_size, rotation=angle)
    return self.output
