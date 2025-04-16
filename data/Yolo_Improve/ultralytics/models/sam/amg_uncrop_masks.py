def uncrop_masks(masks: torch.Tensor, crop_box: List[int], orig_h: int,
    orig_w: int) ->torch.Tensor:
    """Uncrop masks by padding them to the original image size."""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = x0, pad_x - x0, y0, pad_y - y0
    return torch.nn.functional.pad(masks, pad, value=0)
