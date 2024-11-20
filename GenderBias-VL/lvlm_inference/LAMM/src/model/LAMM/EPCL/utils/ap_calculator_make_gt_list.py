def make_gt_list(self, gt_box_corners, gt_box_sem_cls_labels, gt_box_present):
    batch_gt_map_cls = []
    bsize = gt_box_corners.shape[0]
    for i in range(bsize):
        batch_gt_map_cls.append([(gt_box_sem_cls_labels[i, j].item(),
            gt_box_corners[i, j]) for j in range(gt_box_corners.shape[1]) if
            gt_box_present[i, j] == 1])
    return batch_gt_map_cls
