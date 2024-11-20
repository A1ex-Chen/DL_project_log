@staticmethod
def draw_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0),
    font_scale=1, font_thickness=2, text_color=(0, 255, 0), text_color_bg=(
    0, 0, 0)):
    offset = 5, 5
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(x - y for x, y in zip(pos, offset))
    rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
    cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font,
        font_scale, text_color, font_thickness, cv2.LINE_AA)
    return text_size
