def _mask_image_to_bboxes(self, mask_image: Tensor, mask_colors: List[int]
    ) ->Tensor:
    bboxes = []
    for mask_color in mask_colors:
        pos = (mask_image == mask_color).nonzero()
        if pos.shape[0] > 0:
            left = pos[:, 1].min().item()
            top = pos[:, 0].min().item()
            right = pos[:, 1].max().item()
            bottom = pos[:, 0].max().item()
            bboxes.append([left, top, right, bottom])
    bboxes = torch.tensor(bboxes, dtype=torch.float)
    return bboxes
