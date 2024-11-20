def transform_proposals(dataset_dict, image_shape, transforms, *,
    proposal_topk, min_box_size=0):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        proposal_topk (int): only keep top-K scoring proposals
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if 'proposal_boxes' in dataset_dict:
        boxes = transforms.apply_box(BoxMode.convert(dataset_dict.pop(
            'proposal_boxes'), dataset_dict.pop('proposal_bbox_mode'),
            BoxMode.XYXY_ABS))
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(dataset_dict.pop(
            'proposal_objectness_logits').astype('float32'))
        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_size)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]
        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict['proposals'] = proposals
