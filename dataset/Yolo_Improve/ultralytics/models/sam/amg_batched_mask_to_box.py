def batched_mask_to_box(masks: torch.Tensor) ->torch.Tensor:
    """
    Calculates boxes in XYXY format around masks.

    Return [0,0,0,0] for an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[
        None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * ~in_height
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[
        None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * ~in_width
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges],
        dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
