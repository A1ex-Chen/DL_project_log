@torch.no_grad()
def label_and_sample_anchors(self, anchors: List[RotatedBoxes],
    gt_instances: List[Instances]):
    """
        Args:
            anchors (list[RotatedBoxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across feature maps. Label values are in {-1, 0, 1},
                with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            list[Tensor]:
                i-th element is a Nx5 tensor, where N is the total number of anchors across
                feature maps.  The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as 1.
        """
    anchors = RotatedBoxes.cat(anchors)
    gt_boxes = [x.gt_boxes for x in gt_instances]
    del gt_instances
    gt_labels = []
    matched_gt_boxes = []
    for gt_boxes_i in gt_boxes:
        """
            gt_boxes_i: ground-truth boxes for i-th image
            """
        match_quality_matrix = retry_if_cuda_oom(pairwise_iou_rotated)(
            gt_boxes_i, anchors)
        matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(
            match_quality_matrix)
        gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
        gt_labels_i = self._subsample_labels(gt_labels_i)
        if len(gt_boxes_i) == 0:
            matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
        else:
            matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
        gt_labels.append(gt_labels_i)
        matched_gt_boxes.append(matched_gt_boxes_i)
    return gt_labels, matched_gt_boxes
