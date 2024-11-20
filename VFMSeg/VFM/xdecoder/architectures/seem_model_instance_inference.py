def instance_inference(self, mask_cls, mask_pred, box_pred):
    image_size = mask_pred.shape[-2:]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(self.sem_seg_head.num_classes, device=self.device
        ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.
        test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]
    topk_indices = topk_indices // self.sem_seg_head.num_classes
    mask_pred = mask_pred[topk_indices]
    if box_pred is not None:
        box_pred = box_pred[topk_indices]
    if self.panoptic_on:
        keep = torch.zeros_like(scores_per_image).bool()
        for i, lab in enumerate(labels_per_image):
            keep[i
                ] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values(
                )
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]
        if box_pred is not None:
            box_pred = box_pred[keep]
    result = Instances(image_size)
    result.pred_masks = (mask_pred > 0).float()
    if box_pred is not None:
        result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
    else:
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.
        pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1
        ) + 1e-06)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result
