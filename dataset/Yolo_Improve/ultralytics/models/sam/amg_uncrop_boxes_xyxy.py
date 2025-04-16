def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) ->torch.Tensor:
    """Uncrop bounding boxes by adding the crop box offset."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset
