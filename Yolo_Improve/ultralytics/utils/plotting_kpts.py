def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True, conf_thres=0.25
    ):
    """
        Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """
    if self.pil:
        self.im = np.asarray(self.im).copy()
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim in {2, 3}
    kpt_line &= is_pose
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < conf_thres:
                    continue
            cv2.circle(self.im, (int(x_coord), int(y_coord)), radius,
                color_k, -1, lineType=cv2.LINE_AA)
    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(self.skeleton):
            pos1 = int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1])
            pos2 = int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1])
            if ndim == 3:
                conf1 = kpts[sk[0] - 1, 2]
                conf2 = kpts[sk[1] - 1, 2]
                if conf1 < conf_thres or conf2 < conf_thres:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0
                ] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0
                ] < 0 or pos2[1] < 0:
                continue
            cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[
                i]], thickness=2, lineType=cv2.LINE_AA)
    if self.pil:
        self.fromarray(self.im)
