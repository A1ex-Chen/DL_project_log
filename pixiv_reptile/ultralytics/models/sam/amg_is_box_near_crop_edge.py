def is_box_near_crop_edge(boxes: torch.Tensor, crop_box: List[int],
    orig_box: List[int], atol: float=20.0) ->torch.Tensor:
    """Return a boolean tensor indicating if boxes are near the crop edge."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=
        boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=
        boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=
        atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=
        atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)
