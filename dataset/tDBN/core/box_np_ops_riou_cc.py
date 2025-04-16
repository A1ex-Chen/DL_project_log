def riou_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    boxes_corners = center_to_corner_box2d(rbboxes[:, :2], rbboxes[:, 2:4],
        rbboxes[:, 4])
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(qrbboxes[:, :2], qrbboxes[:, 2:
        4], qrbboxes[:, 4])
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return box_ops_cc.rbbox_iou(boxes_corners, qboxes_corners, standup_iou,
        standup_thresh)
