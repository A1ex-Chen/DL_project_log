def crop_and_resize(image_batch: Tensor, norm_crop_center_x_batch: Tensor,
    norm_crop_center_y_batch: Tensor, norm_crop_width_batch: Tensor,
    norm_crop_height_batch: Tensor, resized_width: int, resized_height: int
    ) ->Tensor:
    assert image_batch.ndim == 4
    assert norm_crop_center_x_batch.ndim == 1
    assert norm_crop_center_y_batch.ndim == 1
    assert norm_crop_width_batch.ndim == 1
    assert norm_crop_height_batch.ndim == 1
    assert ((norm_crop_center_x_batch >= 0) & (norm_crop_center_x_batch <= 1)
        ).all().item()
    assert ((norm_crop_center_y_batch >= 0) & (norm_crop_center_y_batch <= 1)
        ).all().item()
    assert ((norm_crop_width_batch >= 0) & (norm_crop_width_batch <= 1)).all(
        ).item()
    assert ((norm_crop_height_batch >= 0) & (norm_crop_height_batch <= 1)).all(
        ).item()
    batch_size, _, image_height, image_width = image_batch.shape
    resized_crop_batch = []
    for b in range(batch_size):
        image = image_batch[b]
        norm_crop_center_x = norm_crop_center_x_batch[b]
        norm_crop_center_y = norm_crop_center_y_batch[b]
        norm_crop_width = norm_crop_width_batch[b]
        norm_crop_height = norm_crop_height_batch[b]
        crop_width = int(image_width * norm_crop_width)
        crop_height = int(image_height * norm_crop_height)
        norm_crop_left = norm_crop_center_x - norm_crop_width / 2
        norm_crop_top = norm_crop_center_y - norm_crop_height / 2
        x_samples = torch.linspace(start=0, end=1, steps=crop_width).to(
            norm_crop_width) * norm_crop_width + norm_crop_left
        y_samples = torch.linspace(start=0, end=1, steps=crop_height).to(
            norm_crop_height) * norm_crop_height + norm_crop_top
        grid = torch.meshgrid(x_samples, y_samples)
        grid = torch.stack(grid, dim=-1)
        grid = grid.transpose(0, 1)
        grid = grid * 2 - 1
        crop_batch = F.grid_sample(input=image.unsqueeze(dim=0), grid=grid.
            unsqueeze(dim=0), mode='bilinear', align_corners=True)
        resized_crop = F.interpolate(input=crop_batch, size=(resized_height,
            resized_width), mode='bilinear', align_corners=True).squeeze(dim=0)
        resized_crop_batch.append(resized_crop)
    resized_crop_batch = torch.stack(resized_crop_batch, dim=0)
    return resized_crop_batch
