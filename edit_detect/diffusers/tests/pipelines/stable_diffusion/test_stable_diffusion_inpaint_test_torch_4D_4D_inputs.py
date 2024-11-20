def test_torch_4D_4D_inputs(self):
    height, width = 32, 32
    im_tensor = torch.randint(0, 255, (1, 3, height, width), dtype=torch.uint8)
    mask_tensor = torch.randint(0, 255, (1, 1, height, width), dtype=torch.
        uint8) > 127.5
    im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
    mask_np = mask_tensor.numpy()[0][0]
    t_mask_tensor, t_masked_tensor, t_image_tensor = (
        prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor,
        height, width, return_image=True))
    t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(im_np,
        mask_np, height, width, return_image=True)
    self.assertTrue((t_mask_tensor == t_mask_np).all())
    self.assertTrue((t_masked_tensor == t_masked_np).all())
    self.assertTrue((t_image_tensor == t_image_np).all())
