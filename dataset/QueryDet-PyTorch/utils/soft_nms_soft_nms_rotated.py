def soft_nms_rotated(boxes, scores, method, gaussian_sigma,
    linear_threshold, prune_threshold):
    """
    Performs soft non-maximum suppression algorithm on rotated boxes

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
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
            [1]: float tensor with the re-scored scores of the elements that were kept    """
    return _soft_nms(RotatedBoxes, pairwise_iou_rotated, boxes, scores,
        method, gaussian_sigma, linear_threshold, prune_threshold)
