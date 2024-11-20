def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    cv2.rectangle(img, bbox, (255, 0, 255), rt)
    cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
    cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    return img
