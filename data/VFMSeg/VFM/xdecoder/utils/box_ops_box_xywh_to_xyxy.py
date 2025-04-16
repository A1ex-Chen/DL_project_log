def box_xywh_to_xyxy(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, x0 + x1, y0 + y1]
    return torch.stack(b, dim=-1)
