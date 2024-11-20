@torch.no_grad()
def label_anchors(self, anchors, gt_instances):
    """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """
    anchors = Boxes.cat(anchors)
    gt_labels = []
    matched_gt_boxes = []
    for gt_per_image in gt_instances:
        match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
        matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
        del match_quality_matrix
        if len(gt_per_image) > 0:
            matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
            gt_labels_i = gt_per_image.gt_classes[matched_idxs]
            gt_labels_i[anchor_labels == 0] = self.num_classes
            gt_labels_i[anchor_labels == -1] = -1
        else:
            matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes
        gt_labels.append(gt_labels_i)
        matched_gt_boxes.append(matched_gt_boxes_i)
    return gt_labels, matched_gt_boxes
