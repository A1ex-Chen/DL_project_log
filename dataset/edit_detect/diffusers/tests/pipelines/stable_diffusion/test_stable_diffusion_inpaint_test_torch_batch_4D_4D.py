def test_torch_batch_4D_4D(self):
    height, width = 32, 32
    im_tensor = torch.randint(0, 255, (2, 3, height, width), dtype=torch.uint8)
    mask_tensor = torch.randint(0, 255, (2, 1, height, width), dtype=torch.
        uint8) > 127.5
    im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
    mask_nps = [mask.numpy()[0] for mask in mask_tensor]
    t_mask_tensor, t_masked_tensor, t_image_tensor = (
        prepare_mask_and_masked_image(im_tensor / 127.5 - 1, mask_tensor,
        height, width, return_image=True))
    nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=
        True) for i, m in zip(im_nps, mask_nps)]
    t_mask_np = torch.cat([n[0] for n in nps])
    t_masked_np = torch.cat([n[1] for n in nps])
    t_image_np = torch.cat([n[2] for n in nps])
    self.assertTrue((t_mask_tensor == t_mask_np).all())
    self.assertTrue((t_masked_tensor == t_masked_np).all())
    self.assertTrue((t_image_tensor == t_image_np).all())
