@staticmethod
def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128),
    txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] +
            h + 2), font, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA
            )
