def sanitize_batch(batch, dataset_info):
    """Sanitizes input batch for inference, ensuring correct format and dimensions."""
    batch['cls'] = batch['cls'].flatten().int().tolist()
    box_cls_pair = sorted(zip(batch['bboxes'].tolist(), batch['cls']), key=
        lambda x: x[1])
    batch['bboxes'] = [box for box, _ in box_cls_pair]
    batch['cls'] = [cls for _, cls in box_cls_pair]
    batch['labels'] = [dataset_info['names'][i] for i in batch['cls']]
    batch['masks'] = batch['masks'].tolist() if 'masks' in batch else [[[]]]
    batch['keypoints'] = batch['keypoints'].tolist(
        ) if 'keypoints' in batch else [[[]]]
    return batch
