def nms_cc(dets, thresh):
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    return non_max_suppression_cpu(dets, order, thresh, 1.0)
