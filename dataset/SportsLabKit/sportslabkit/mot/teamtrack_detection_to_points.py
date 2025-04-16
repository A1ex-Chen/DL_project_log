def detection_to_points(self, detection, H):
    box = detection.box
    bcx, bcy = box[0] + box[2] / 2, box[1] + box[3]
    return cv2.perspectiveTransform(np.array([[[bcx, bcy]]], dtype=
        'float32'), H)[0][0]
