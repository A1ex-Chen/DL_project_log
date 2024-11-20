@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[
    Instances], vis_period: int=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3
        ), 'Mask prediction must be square!'
    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=
                torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len).to(device
            =pred_mask_logits.device)
        gt_masks.append(gt_masks_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - mask_incorrect.sum().item() / max(mask_incorrect.
        numel(), 1.0)
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0)
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(
        num_positive, 1.0)
    storage = get_event_storage()
    storage.put_scalar('mask_rcnn/accuracy', mask_accuracy)
    storage.put_scalar('mask_rcnn/false_positive', false_positive)
    storage.put_scalar('mask_rcnn/false_negative', false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = 'Left: mask prediction;   Right: mask GT'
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f' ({idx})', vis_mask)
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits,
        gt_masks, reduction='mean')
    return mask_loss
