def vis(self, img, boxes, scores, cls_ids):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < self.confThreshold:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        text = '{}:{:.2f}'.format(self.class_names[cls_id], score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size, baseline = cv2.getTextSize(text, font, 0.7, 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 178, 50), 2)
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 +
            txt_size[1] + baseline), (0, 0, 0), -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.7, (0, 255, 
            255), 1, cv2.LINE_AA)
    return img
