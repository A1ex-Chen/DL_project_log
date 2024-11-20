def forward(self, padded_image_batch: Tensor, gt_bboxes_batch: List[Tensor]
    =None, padded_gt_masks_batch: List[Tensor]=None, gt_classes_batch: List
    [Tensor]=None) ->Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
    if self.training:
        padded_image_batch = [it for it in padded_image_batch]
        targets = []
        for gt_bboxes, gt_classes, padded_gt_masks in zip(gt_bboxes_batch,
            gt_classes_batch, padded_gt_masks_batch):
            target = {'boxes': gt_bboxes, 'labels': gt_classes, 'masks':
                padded_gt_masks}
            targets.append(target)
        out = self.net(padded_image_batch, targets)
        return out['loss_objectness'], out['loss_rpn_box_reg'], out[
            'loss_classifier'], out['loss_box_reg'], out['loss_mask']
    else:
        padded_image_batch = [it for it in padded_image_batch]
        out_list = self.net(padded_image_batch)
        detection_bboxes_batch = [out['boxes'] for out in out_list]
        detection_classes_batch = [out['labels'] for out in out_list]
        detection_probs_batch = [out['scores'] for out in out_list]
        detection_masks_batch = [out['masks'] for out in out_list]
        return (detection_bboxes_batch, detection_classes_batch,
            detection_probs_batch, detection_masks_batch)
