def yolov6_decode(feats: List[ndarray], conf_thres: float, iou_thres: float,
    num_labels: int=80, **kwargs):
    proposal_boxes: List[ndarray] = []
    proposal_scores: List[float] = []
    proposal_labels: List[int] = []
    for i, feat in enumerate(feats):
        feat = np.ascontiguousarray(feat.transpose((1, 2, 0)))
        stride = 8 << i
        score_feat, box_feat = np.split(feat, [num_labels], -1)
        _argmax = score_feat.argmax(-1)
        _max = score_feat.max(-1)
        indices = np.where(_max > conf_thres)
        hIdx, wIdx = indices
        num_proposal = hIdx.size
        if not num_proposal:
            continue
        scores = _max[hIdx, wIdx]
        boxes = box_feat[hIdx, wIdx]
        labels = _argmax[hIdx, wIdx]
        for k in range(num_proposal):
            score = scores[k]
            label = labels[k]
            x0, y0, x1, y1 = boxes[k]
            x0 = (wIdx[k] + 0.5 - x0) * stride
            y0 = (hIdx[k] + 0.5 - y0) * stride
            x1 = (wIdx[k] + 0.5 + x1) * stride
            y1 = (hIdx[k] + 0.5 + y1) * stride
            w = x1 - x0
            h = y1 - y0
            proposal_scores.append(float(score))
            proposal_boxes.append(np.array([x0, y0, w, h], dtype=np.float32))
            proposal_labels.append(int(label))
    if MINOR >= 7:
        indices = cv2.dnn.NMSBoxesBatched(proposal_boxes, proposal_scores,
            proposal_labels, conf_thres, iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores,
            conf_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores,
            conf_thres, iou_thres).flatten()
    if not len(indices):
        return [], [], []
    nmsd_boxes: List[ndarray] = []
    nmsd_scores: List[float] = []
    nmsd_labels: List[int] = []
    for idx in indices:
        box = proposal_boxes[idx]
        box[2:] = box[:2] + box[2:]
        score = proposal_scores[idx]
        label = proposal_labels[idx]
        nmsd_boxes.append(box)
        nmsd_scores.append(score)
        nmsd_labels.append(label)
    return nmsd_boxes, nmsd_scores, nmsd_labels
