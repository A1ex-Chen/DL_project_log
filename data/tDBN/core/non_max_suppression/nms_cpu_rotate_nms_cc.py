def rotate_nms_cc(dets, thresh):
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2
        :4], dets[:, 4])
    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)
    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou,
        thresh)
