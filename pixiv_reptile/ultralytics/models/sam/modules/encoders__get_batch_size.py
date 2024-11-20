def _get_batch_size(self, points: Optional[Tuple[torch.Tensor, torch.Tensor
    ]], boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) ->int:
    """Gets the batch size of the output given the batch size of the input prompts."""
    if points is not None:
        return points[0].shape[0]
    elif boxes is not None:
        return boxes.shape[0]
    elif masks is not None:
        return masks.shape[0]
    else:
        return 1
