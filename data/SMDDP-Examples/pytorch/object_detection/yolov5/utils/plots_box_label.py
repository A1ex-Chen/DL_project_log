def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 
    255, 255)):
    if self.pil or not is_ascii(label):
        self.draw.rectangle(box, width=self.lw, outline=color)
        if label:
            w, h = self.font.getsize(label)
            outside = box[1] - h >= 0
            self.draw.rectangle((box[0], box[1] - h if outside else box[1],
                box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1),
                fill=color)
            self.draw.text((box[0], box[1] - h if outside else box[1]),
                label, fill=txt_color, font=self.font)
    else:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=
            cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3,
                thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else 
                p1[1] + h + 2), 0, self.lw / 3, txt_color, thickness=tf,
                lineType=cv2.LINE_AA)
