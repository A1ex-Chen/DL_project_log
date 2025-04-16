def clip(self, box_size: Tuple[int, int], clip_angle_threshold: float=1.0
    ) ->None:
    """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        For RRPN:
        Only clip boxes that are almost horizontal with a tolerance of
        clip_angle_threshold to maintain backward compatibility.

        Rotated boxes beyond this threshold are not clipped for two reasons:

        1. There are potentially multiple ways to clip a rotated box to make it
           fit within the image.
        2. It's tricky to make the entire rectangular box fit within the image
           and still be able to not leave out pixels of interest.

        Therefore we rely on ops like RoIAlignRotated to safely handle this.

        Args:
            box_size (height, width): The clipping box's size.
            clip_angle_threshold:
                Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
                we do the clipping as horizontal boxes.
        """
    h, w = box_size
    self.normalize_angles()
    idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]
    x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0
    y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0
    x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0
    y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0
    x1.clamp_(min=0, max=w)
    y1.clamp_(min=0, max=h)
    x2.clamp_(min=0, max=w)
    y2.clamp_(min=0, max=h)
    self.tensor[idx, 0] = (x1 + x2) / 2.0
    self.tensor[idx, 1] = (y1 + y2) / 2.0
    self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)
    self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)
