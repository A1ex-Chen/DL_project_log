def _cv2_rotate(self, im):
    if self.orientation == 0:
        return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif self.orientation == 180:
        return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif self.orientation == 90:
        return cv2.rotate(im, cv2.ROTATE_180)
    return im
