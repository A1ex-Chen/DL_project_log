def _soft_nms(box_class, pairwise_iou_func, boxes, scores, method,
    gaussian_sigma, linear_threshold, prune_threshold):
    """
    Soft non-max suppression algorithm.

    Implementation of [Soft-NMS -- Improving Object Detection With One Line of Codec]
    (https://arxiv.org/abs/1704.04503)

    Args:
        box_class (cls): one of Box, RotatedBoxes
        pairwise_iou_func (func): one of pairwise_iou, pairwise_iou_rotated
        boxes (Tensor[N, ?]):
           boxes where NMS will be performed
           if Boxes, in (x1, y1, x2, y2) format
           if RotatedBoxes, in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]

    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(scores.size()[0])
    idxs_out = []
    scores_out = []
    while scores.numel() > 0:
        top_idx = torch.argmax(scores)
        idxs_out.append(idxs[top_idx].item())
        scores_out.append(scores[top_idx].item())
        top_box = boxes[top_idx]
        ious = pairwise_iou_func(box_class(top_box.unsqueeze(0)), box_class
            (boxes))[0]
        if method == 'linear':
            decay = torch.ones_like(ious)
            decay_mask = ious > linear_threshold
            decay[decay_mask] = 1 - ious[decay_mask]
        elif method == 'gaussian':
            decay = torch.exp(-torch.pow(ious, 2) / gaussian_sigma)
        elif method == 'hard':
            decay = (ious < linear_threshold).float()
        else:
            raise NotImplementedError('{} soft nms method not implemented.'
                .format(method))
        scores *= decay
        keep = scores > prune_threshold
        keep[top_idx] = False
        boxes = boxes[keep]
        scores = scores[keep]
        idxs = idxs[keep]
    return torch.tensor(idxs_out).to(boxes.device), torch.tensor(scores_out
        ).to(scores.device)
