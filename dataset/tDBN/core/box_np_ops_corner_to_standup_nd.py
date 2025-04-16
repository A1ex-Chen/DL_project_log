def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)
