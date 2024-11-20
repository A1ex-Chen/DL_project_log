def accumulate(self, batch_pred_map_cls, batch_gt_map_cls):
    """Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
    bsize = len(batch_pred_map_cls)
    assert bsize == len(batch_gt_map_cls)
    for i in range(bsize):
        self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
        self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
        self.scan_cnt += 1
