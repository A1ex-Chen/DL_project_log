def test_tensor_range(self):
    height, width = 32, 32
    with self.assertRaises(ValueError):
        prepare_mask_and_masked_image(torch.ones(3, height, width) * 2,
            torch.rand(height, width), height, width, return_image=True)
    with self.assertRaises(ValueError):
        prepare_mask_and_masked_image(torch.ones(3, height, width) * -2,
            torch.rand(height, width), height, width, return_image=True)
    with self.assertRaises(ValueError):
        prepare_mask_and_masked_image(torch.rand(3, height, width), torch.
            ones(height, width) * 2, height, width, return_image=True)
    with self.assertRaises(ValueError):
        prepare_mask_and_masked_image(torch.rand(3, height, width), torch.
            ones(height, width) * -1, height, width, return_image=True)
