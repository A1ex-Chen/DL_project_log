def process_batch(self, detections, gt_bboxes, gt_cls):
    """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
    if gt_cls.shape[0] == 0:
        if detections is not None:
            detections = detections[detections[:, 4] > self.conf]
            detection_classes = detections[:, 5].int()
            for dc in detection_classes:
                self.matrix[dc, self.nc] += 1
        return
    if detections is None:
        gt_classes = gt_cls.int()
        for gc in gt_classes:
            self.matrix[self.nc, gc] += 1
        return
    detections = detections[detections[:, 4] > self.conf]
    gt_classes = gt_cls.int()
    detection_classes = detections[:, 5].int()
    is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5
    iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections
        [:, -1:]], dim=-1)) if is_obb else box_iou(gt_bboxes, detections[:, :4]
        )
    x = torch.where(iou > self.iou_thres)
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
            ).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))
    n = matches.shape[0] > 0
    m0, m1, _ = matches.transpose().astype(int)
    for i, gc in enumerate(gt_classes):
        j = m0 == i
        if n and sum(j) == 1:
            self.matrix[detection_classes[m1[j]], gc] += 1
        else:
            self.matrix[self.nc, gc] += 1
    if n:
        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1
