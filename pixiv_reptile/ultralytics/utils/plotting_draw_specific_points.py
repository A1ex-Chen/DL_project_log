def draw_specific_points(self, keypoints, indices=None, shape=(640, 640),
    radius=2, conf_thres=0.25):
    """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): list of keypoints data to be plotted
            indices (list): keypoints ids list to be plotted
            shape (tuple): imgsz for model inference
            radius (int): Keypoint radius value
        """
    if indices is None:
        indices = [2, 5, 7]
    for i, k in enumerate(keypoints):
        if i in indices:
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (
                    0, 255, 0), -1, lineType=cv2.LINE_AA)
    return self.im
