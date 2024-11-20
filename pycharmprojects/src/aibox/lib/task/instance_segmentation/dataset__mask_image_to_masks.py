def _mask_image_to_masks(self, mask_image: Tensor, mask_colors: List[int]
    ) ->Tensor:
    mask_colors = torch.tensor(mask_colors, dtype=torch.uint8)
    masks = mask_image.repeat(mask_colors.shape[0], 1, 1)
    masks = (masks == mask_colors.view(-1, 1, 1).expand_as(masks)).type_as(
        masks)
    return masks
