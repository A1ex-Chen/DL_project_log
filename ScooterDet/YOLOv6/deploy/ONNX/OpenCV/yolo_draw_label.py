def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] +
        baseline), BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE,
        FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
