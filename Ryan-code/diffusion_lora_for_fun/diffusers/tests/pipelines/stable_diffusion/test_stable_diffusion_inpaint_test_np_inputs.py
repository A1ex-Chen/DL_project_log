def test_np_inputs(self):
    height, width = 32, 32
    im_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    im_pil = Image.fromarray(im_np)
    mask_np = np.random.randint(0, 255, (height, width), dtype=np.uint8
        ) > 127.5
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(im_np,
        mask_np, height, width, return_image=True)
    t_mask_pil, t_masked_pil, t_image_pil = prepare_mask_and_masked_image(
        im_pil, mask_pil, height, width, return_image=True)
    self.assertTrue((t_mask_np == t_mask_pil).all())
    self.assertTrue((t_masked_np == t_masked_pil).all())
    self.assertTrue((t_image_np == t_image_pil).all())
