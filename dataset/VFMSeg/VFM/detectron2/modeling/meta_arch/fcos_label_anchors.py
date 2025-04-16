@torch.no_grad()
def label_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
    """
        Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS
        anchor matching rule.

        Unlike RetinaNet, there are no ignored anchors.
        """
    gt_labels, matched_gt_boxes = [], []
    for inst in gt_instances:
        if len(inst) > 0:
            match_quality_matrix = self._match_anchors(inst.gt_boxes, anchors)
            match_quality, matched_idxs = match_quality_matrix.max(dim=0)
            matched_idxs[match_quality < 1e-05] = -1
            matched_gt_boxes_i = inst.gt_boxes.tensor[matched_idxs.clip(min=0)]
            gt_labels_i = inst.gt_classes[matched_idxs.clip(min=0)]
            gt_labels_i[matched_idxs < 0] = self.num_classes
        else:
            matched_gt_boxes_i = torch.zeros_like(Boxes.cat(anchors).tensor)
            gt_labels_i = torch.full((len(matched_gt_boxes_i),), fill_value
                =self.num_classes, dtype=torch.long, device=
                matched_gt_boxes_i.device)
        gt_labels.append(gt_labels_i)
        matched_gt_boxes.append(matched_gt_boxes_i)
    return gt_labels, matched_gt_boxes
