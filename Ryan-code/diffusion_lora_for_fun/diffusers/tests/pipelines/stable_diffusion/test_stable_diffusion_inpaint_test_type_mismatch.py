def test_type_mismatch(self):
    height, width = 32, 32
    with self.assertRaises(TypeError):
        prepare_mask_and_masked_image(torch.rand(3, height, width), torch.
            rand(3, height, width).numpy(), height, width, return_image=True)
    with self.assertRaises(TypeError):
        prepare_mask_and_masked_image(torch.rand(3, height, width).numpy(),
            torch.rand(3, height, width), height, width, return_image=True)
