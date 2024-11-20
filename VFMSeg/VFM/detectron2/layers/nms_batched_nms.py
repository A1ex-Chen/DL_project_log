def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.
    Tensor, iou_threshold: float):
    """
    Same as torchvision.ops.boxes.batched_nms, but with float().
    """
    assert boxes.shape[-1] == 4
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)
