@staticmethod
def postprocess(preds: torch.Tensor, max_det: int, nc: int=80):
    """
        Post-processes the predictions obtained from a YOLOv10 model.

        Args:
            preds (torch.Tensor): The predictions obtained from the model. It should have a shape of (batch_size, num_boxes, 4 + num_classes).
            max_det (int): The maximum number of detections to keep.
            nc (int, optional): The number of classes. Defaults to 80.

        Returns:
            (torch.Tensor): The post-processed predictions with shape (batch_size, max_det, 6),
                including bounding boxes, scores and cls.
        """
    assert 4 + nc == preds.shape[-1]
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, min(max_det, max_scores.
        shape[1]), axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape
        [-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.
        shape[-1]))
    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1,
        boxes.shape[-1]))
    return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(
        boxes.dtype)], dim=-1)
