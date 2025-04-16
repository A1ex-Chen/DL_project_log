def test_crop_and_resize():
    image_batch = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [
        3, 3, 4, 4]], dtype=torch.float, requires_grad=True).unsqueeze(dim=0
        ).unsqueeze(dim=0)
    norm_crop_center_x_batch = torch.tensor([0.625], dtype=torch.float,
        requires_grad=True)
    norm_crop_center_y_batch = torch.tensor([0.75], dtype=torch.float,
        requires_grad=True)
    norm_crop_width_batch = torch.tensor([0.75], dtype=torch.float,
        requires_grad=True)
    norm_crop_height_batch = torch.tensor([0.5], dtype=torch.float,
        requires_grad=True)
    resized_crop_batch = crop_and_resize(image_batch,
        norm_crop_center_x_batch, norm_crop_center_y_batch,
        norm_crop_width_batch, norm_crop_height_batch, resized_width=4,
        resized_height=4)
    print('image_batch:\n', image_batch)
    print('resized_crop_batch:\n', resized_crop_batch)
    image_batch.retain_grad()
    norm_crop_center_x_batch.retain_grad()
    norm_crop_center_y_batch.retain_grad()
    norm_crop_width_batch.retain_grad()
    norm_crop_height_batch.retain_grad()
    resized_crop_batch.retain_grad()
    resized_crop_batch.sum().backward()
    print('image_batch.grad:\n', image_batch.grad)
    print('norm_crop_center_x_batch.grad:\n', norm_crop_center_x_batch.grad)
    print('norm_crop_center_y_batch.grad:\n', norm_crop_center_y_batch.grad)
    print('norm_crop_width_batch.grad:\n', norm_crop_width_batch.grad)
    print('norm_crop_height_batch.grad:\n', norm_crop_height_batch.grad)
    print('resized_crop_batch.grad:\n', resized_crop_batch.grad)
