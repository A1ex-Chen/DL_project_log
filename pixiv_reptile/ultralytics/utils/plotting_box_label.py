def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 
    255, 255), rotated=False):
    """
        Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (R, G, B).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Variable used to check if task is OBB
        """
    txt_color = self.get_txt_color(color, txt_color)
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    if self.pil or not is_ascii(label):
        if rotated:
            p1 = box[0]
            self.draw.polygon([tuple(b) for b in box], width=self.lw,
                outline=color)
        else:
            p1 = box[0], box[1]
            self.draw.rectangle(box, width=self.lw, outline=color)
        if label:
            w, h = self.font.getsize(label)
            outside = p1[1] >= h
            if p1[0] > self.im.size[0] - w:
                p1 = self.im.size[0] - w, p1[1]
            self.draw.rectangle((p1[0], p1[1] - h if outside else p1[1], p1
                [0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1), fill
                =color)
            self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label,
                fill=txt_color, font=self.font)
    else:
        if rotated:
            p1 = [int(b) for b in box[0]]
            cv2.polylines(self.im, [np.asarray(box, dtype=int)], True,
                color, self.lw)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw,
                lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=
                self.tf)[0]
            h += 3
            outside = p1[1] >= h
            if p1[0] > self.im.shape[1] - w:
                p1 = self.im.shape[1] - w, p1[1]
            p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else 
                p1[1] + h - 1), 0, self.sf, txt_color, thickness=self.tf,
                lineType=cv2.LINE_AA)
