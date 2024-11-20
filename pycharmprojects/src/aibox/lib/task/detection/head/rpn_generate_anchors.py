def generate_anchors(self, padded_image_width: int, padded_image_height:
    int, num_x_anchors: int, num_y_anchors: int, scale: float) ->Tensor:
    center_ys = torch.linspace(start=0, end=padded_image_height, steps=
        num_y_anchors + 2)[1:-1]
    center_xs = torch.linspace(start=0, end=padded_image_width, steps=
        num_x_anchors + 2)[1:-1]
    ratios = torch.tensor(self._anchor_ratios, dtype=torch.float)
    ratios = ratios[:, 0] / ratios[:, 1]
    sizes = torch.tensor(self._anchor_sizes, dtype=torch.float)
    center_ys, center_xs, ratios, sizes = torch.meshgrid(center_ys,
        center_xs, ratios, sizes)
    center_ys = center_ys.reshape(-1)
    center_xs = center_xs.reshape(-1)
    ratios = ratios.reshape(-1)
    sizes = sizes.reshape(-1)
    widths = sizes * torch.sqrt(1 / ratios) * scale
    heights = sizes * torch.sqrt(ratios) * scale
    center_based_anchor_bboxes = torch.stack((center_xs, center_ys, widths,
        heights), dim=1)
    anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
    return anchor_bboxes
