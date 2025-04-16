@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0])
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1])
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]
                                ) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]
                                ) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] -
                                    A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] -
                                    A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        box_overlap_qbox = True
                        for l in range(4):
                            for k in range(4):
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j,
                                    l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[
                                    j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break
                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):
                                for k in range(4):
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] -
                                        boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] -
                                        boxes[i, l, 1])
                                    if cross >= 0:
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True
                        else:
                            ret[i, j] = True
    return ret
