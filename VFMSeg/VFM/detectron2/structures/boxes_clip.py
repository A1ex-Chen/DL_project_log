def clip(self, box_size: Tuple[int, int]) ->None:
    """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
    assert torch.isfinite(self.tensor).all(
        ), 'Box tensor contains infinite or NaN!'
    h, w = box_size
    x1 = self.tensor[:, 0].clamp(min=0, max=w)
    y1 = self.tensor[:, 1].clamp(min=0, max=h)
    x2 = self.tensor[:, 2].clamp(min=0, max=w)
    y2 = self.tensor[:, 3].clamp(min=0, max=h)
    self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)
